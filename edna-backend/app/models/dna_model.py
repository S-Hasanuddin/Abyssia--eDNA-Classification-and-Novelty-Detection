"""
dna_model.py — Model architecture definitions
================================================
CRITICAL: The projection head includes nn.Dropout(0.5) between the two
linear layers, matching phase3_finetune.py and ft_final.pt.

The inference_api.py shipped with the project OMITS this Dropout,
which causes a state_dict key mismatch when loading ft_final.pt.
This file corrects that.

Architecture (96M parameters total):
  - Encoder: 12-layer BERT, 12 heads, 768-dim  (~87.5M params)
  - Bottom-up classifier: Family→Order→Class→Phylum→Kingdom→Domain
  - Projection head: 768→512→Dropout(0.5)→256  (~525K params)
  - 100 learnable prototypes (256-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from typing import Dict


class DNATransformerEncoder(nn.Module):
    """BERT-style encoder for 6-mer tokenized DNA sequences."""

    def __init__(self, vocab_size: int = 4101):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]  # CLS token


class BottomUpTaxonomyClassifier(nn.Module):
    """
    Bottom-up cascaded classifier.
    Family is predicted first from the raw embedding; each subsequent
    (coarser) rank receives the embedding concatenated with the softmax
    probabilities of the previous (finer) rank.
    """

    def __init__(self, hidden_size: int, num_classes_per_rank: Dict[str, int],
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.family_classifier = nn.Linear(
            hidden_size, num_classes_per_rank['family'])
        self.order_classifier = nn.Linear(
            hidden_size + num_classes_per_rank['family'],
            num_classes_per_rank['order'])
        self.class_classifier = nn.Linear(
            hidden_size + num_classes_per_rank['order'],
            num_classes_per_rank['class'])
        self.phylum_classifier = nn.Linear(
            hidden_size + num_classes_per_rank['class'],
            num_classes_per_rank['phylum'])
        self.kingdom_classifier = nn.Linear(
            hidden_size + num_classes_per_rank['phylum'],
            num_classes_per_rank['kingdom'])
        self.domain_classifier = nn.Linear(
            hidden_size + num_classes_per_rank['kingdom'],
            num_classes_per_rank['domain'])

    def forward(self, embeddings):
        e = self.dropout(embeddings)
        logits, probs = {}, {}
        logits['family'] = self.family_classifier(e)
        probs['family'] = F.softmax(logits['family'], dim=1)
        logits['order'] = self.order_classifier(
            torch.cat([e, probs['family']], dim=1))
        probs['order'] = F.softmax(logits['order'], dim=1)
        logits['class'] = self.class_classifier(
            torch.cat([e, probs['order']], dim=1))
        probs['class'] = F.softmax(logits['class'], dim=1)
        logits['phylum'] = self.phylum_classifier(
            torch.cat([e, probs['class']], dim=1))
        probs['phylum'] = F.softmax(logits['phylum'], dim=1)
        logits['kingdom'] = self.kingdom_classifier(
            torch.cat([e, probs['phylum']], dim=1))
        probs['kingdom'] = F.softmax(logits['kingdom'], dim=1)
        logits['domain'] = self.domain_classifier(
            torch.cat([e, probs['kingdom']], dim=1))
        return logits


class CompleteTaxonomyModel(nn.Module):
    """
    Full model: encoder + bottom-up classifier + projection head + prototypes.

    NOTE: The projection head includes nn.Dropout(0.5) to match
    ft_final.pt / phase3_finetune.py. At eval time, Dropout is a no-op,
    but the layer MUST exist for state_dict keys to align correctly.
    """

    def __init__(self, vocab_size: int, num_classes_per_rank: Dict[str, int]):
        super().__init__()
        self.encoder = DNATransformerEncoder(vocab_size)
        self.classifier = BottomUpTaxonomyClassifier(
            self.encoder.hidden_size, num_classes_per_rank)

        # ── Projection head (MUST include Dropout to match ft_final.pt) ──
        self.projection_head = nn.Sequential(
            nn.Linear(768, 512),     # projection_head.0
            nn.ReLU(),               # projection_head.1
            nn.Dropout(0.5),         # projection_head.2  ← CRITICAL
            nn.Linear(512, 256),     # projection_head.3
        )

        self.num_prototypes = 100
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, 256))

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.encoder(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        projected = F.normalize(self.projection_head(embeddings), dim=1)
        return {
            'embeddings': embeddings,
            'logits': logits,
            'projected': projected,
        }


class EcosystemAnalyzer(nn.Module):
    """
    Sample-level analysis: attention pooling, diversity metrics, ordination.

    NOTE: The attention weights are randomly initialized (never trained in any
    phase). The diversity metrics and PCA ordination are mathematically valid.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # NOTE: The attention_* layers below are kept for checkpoint
        # compatibility (they exist in phase3_finetune's EcosystemAnalyzer)
        # but were NEVER TRAINED — their weights are random.
        # We use deterministic mean pooling instead for sample embeddings.
        self.attention_query = nn.Linear(embedding_dim, embedding_dim)
        self.attention_key = nn.Linear(embedding_dim, embedding_dim)
        self.attention_value = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def sample_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Deterministic mean pooling over all sequence embeddings.

        This replaces the untrained attention_pooling. The attention weights
        (Q, K, V, output_proj) were never trained during any phase, so they
        produce random weighted averages. Mean pooling is deterministic,
        reproducible, and equally valid for generating a sample-level
        representation from per-sequence embeddings.
        """
        return embeddings.mean(dim=0)

    def compute_diversity_metrics(self, logits_dict: Dict,
                                  id_to_label: Dict,
                                  confidence_threshold: float = 0.01) -> Dict:
        """Compute Shannon, Simpson, richness, evenness per rank."""
        metrics = {}
        dominant_taxa = {}
        for rank, preds in logits_dict.items():
            probs = F.softmax(preds, dim=1)
            confs, classes = torch.max(probs, dim=1)
            filtered = classes[confs > confidence_threshold]
            if len(filtered) == 0:
                filtered = classes
            unique, counts = torch.unique(filtered, return_counts=True)
            p = counts.float() / counts.sum()
            shannon = -(p * torch.log(p + 1e-10)).sum().item()
            simpson = 1.0 - (p ** 2).sum().item()
            richness = len(unique)
            evenness = (shannon / torch.log(torch.tensor(float(richness))).item()
                        if richness > 1 else 0.0)
            metrics.update({
                f'{rank}_shannon': round(shannon, 4),
                f'{rank}_simpson': round(simpson, 4),
                f'{rank}_richness': richness,
                f'{rank}_evenness': round(evenness, 4),
            })

            # Dominant taxa: top-5 by abundance
            sorted_idx = torch.argsort(counts, descending=True)[:5]
            rank_labels = id_to_label.get(rank, {})
            dom = []
            total = counts.sum().item()
            for si in sorted_idx:
                idx_val = unique[si].item()
                cnt = counts[si].item()
                taxon = rank_labels.get(idx_val, f'[idx_{idx_val}]')
                dom.append({
                    'taxon': taxon,
                    'count': int(cnt),
                    'relative_abundance': round(cnt / total, 4),
                })
            dominant_taxa[rank] = dom

        return metrics, dominant_taxa

    def compute_ordination(self, embeddings: torch.Tensor) -> list:
        """2D PCA ordination on projected embeddings."""
        if embeddings.size(0) < 2:
            return []
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            coords = torch.mm(centered, Vh[:2].t())
            return [[round(c, 6) for c in row] for row in coords.tolist()]
        except Exception:
            return []

    def analyze(self, projected: torch.Tensor, logits: Dict,
                id_to_label: Dict) -> Dict:
        metrics, dominant_taxa = self.compute_diversity_metrics(
            logits, id_to_label)
        sample_emb = self.sample_pooling(projected).tolist()
        return {
            'diversity_metrics': metrics,
            'dominant_taxa': dominant_taxa,
            'ordination': self.compute_ordination(projected),
            'sample_embedding': [round(x, 6) for x in sample_emb],
        }
