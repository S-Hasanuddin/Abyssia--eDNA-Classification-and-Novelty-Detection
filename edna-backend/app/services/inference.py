"""
inference.py — Core inference pipeline service
================================================
Loads the model, runs classification, novelty detection, and ecosystem analysis.
Uses the corrected model definition from dna_model.py (with Dropout in projection head).

Novel routing fix:
  Previously routed on `any_confident` — since Domain always passes its low
  threshold, nothing ever reached HDBSCAN. Routing is now based on whether the
  most specific confident rank meets MIN_CLASSIFICATION_RANK (default: order).
  Sequences that only resolve to Domain/Kingdom/Phylum/Class are sent to
  novel clustering instead.
"""

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional

from app.config import (
    CHECKPOINT_DIR, CHECKPOINT_PRIORITY, DEVICE, CONFIDENCE_THRESHOLD,
    BATCH_SIZE, HDBSCAN_MIN_CLUSTER, HDBSCAN_MIN_SAMPLES, RANKS,
    RANK_THRESHOLDS, RANK_SPECIFICITY, MIN_CLASSIFICATION_RANK,
)
from app.models.dna_model import CompleteTaxonomyModel, EcosystemAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tokenizer import KmerTokenizer

logger = logging.getLogger(__name__)

# Minimum specificity score a sequence must reach to be classified (not novel)
_MIN_SPECIFICITY = RANK_SPECIFICITY.get(MIN_CLASSIFICATION_RANK.lower(), 4)


def taxonomic_fallback(logits_dict: Dict, id_to_label: Dict,
                       threshold: float = 0.5) -> Dict:
    """
    For a single sequence (batch_size=1 logits), return the most specific
    rank where confidence >= rank-specific threshold (from RANK_THRESHOLDS).
    Falls back toward domain if needed.

    The `threshold` parameter is kept for backward compatibility but is now
    only used as a fallback default for ranks not found in RANK_THRESHOLDS.
    """
    all_ranks = {}
    for rank in RANKS:
        logit = logits_dict[rank]
        if logit.dim() == 2:
            probs = F.softmax(logit, dim=1)[0]
        else:
            probs = F.softmax(logit, dim=0)
        conf, idx = probs.max(dim=0)
        conf = conf.item()
        idx = idx.item()
        taxon = id_to_label.get(rank, {}).get(idx, f'[unknown_idx_{idx}]')

        # Use per-rank threshold; fall back to global threshold if rank missing
        rank_threshold = RANK_THRESHOLDS.get(rank.lower(), threshold)

        all_ranks[rank] = {
            'taxon': taxon,
            'confidence': round(conf, 4),
            'confident': conf >= rank_threshold,
            'threshold_used': rank_threshold,
        }

    # Walk family → domain, pick most specific confident rank
    reported_rank = reported_taxon = reported_confidence = None
    for rank in reversed(RANKS):  # family → domain
        if all_ranks[rank]['confident']:
            reported_rank = rank
            reported_taxon = all_ranks[rank]['taxon']
            reported_confidence = all_ranks[rank]['confidence']
            break

    # Always fall back to domain as a last resort
    if reported_rank is None:
        reported_rank = 'domain'
        reported_taxon = all_ranks['domain']['taxon']
        reported_confidence = all_ranks['domain']['confidence']

    # Specificity score of the reported rank
    reported_specificity = RANK_SPECIFICITY.get(reported_rank, 0)

    return {
        'reported_rank': reported_rank,
        'reported_taxon': reported_taxon,
        'reported_confidence': reported_confidence,
        'fallback_occurred': reported_rank != 'family',
        # True only if confident AND specific enough to classify (not novel)
        'any_confident': any(all_ranks[r]['confident'] for r in RANKS),
        'specific_enough': reported_specificity >= _MIN_SPECIFICITY,
        'reported_specificity': reported_specificity,
        'all_ranks': all_ranks,
    }


class EDNAInferencePipeline:
    """
    Production inference pipeline. Loads once at startup, serves many requests.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'EDNAInferencePipeline':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = DEVICE
        self.threshold = CONFIDENCE_THRESHOLD
        self.rank_thresholds = RANK_THRESHOLDS
        self.min_classification_rank = MIN_CLASSIFICATION_RANK
        self.batch_size = BATCH_SIZE
        self.hdbscan_min_cluster = HDBSCAN_MIN_CLUSTER
        self.hdbscan_min_samples = HDBSCAN_MIN_SAMPLES
        self.tokenizer = KmerTokenizer(k=6)
        self.model = None
        self.analyzer = None
        self.id_to_label = {}
        self.label_to_id = {}
        self.num_classes_per_rank = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """Load model and label maps from checkpoint directory."""
        if self._loaded:
            return

        # ── Find checkpoint ──────────────────────────────────────────────
        ckpt_path = None
        for fname in CHECKPOINT_PRIORITY:
            candidate = CHECKPOINT_DIR / fname
            if candidate.exists():
                ckpt_path = candidate
                break

        if ckpt_path is None:
            logger.warning(
                f"No checkpoint found in {CHECKPOINT_DIR}. "
                f"Looked for: {CHECKPOINT_PRIORITY}. "
                f"API will start but predictions will fail until a checkpoint is provided."
            )
            return

        # ── Load label maps ──────────────────────────────────────────────
        label_map_path = CHECKPOINT_DIR / 'label_to_id.json'
        if not label_map_path.exists():
            logger.warning(f"label_to_id.json not found at {label_map_path}")
            return

        with open(label_map_path) as f:
            self.label_to_id = json.load(f)
        self.id_to_label = {
            rank: {int(v): k for k, v in mapping.items()}
            for rank, mapping in self.label_to_id.items()
        }

        # ── Load model ───────────────────────────────────────────────────
        logger.info(f"Loading checkpoint: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['model_state_dict']

        self.num_classes_per_rank = {
            rank: sd[f'classifier.{rank}_classifier.weight'].shape[0]
            for rank in RANKS
        }

        self.model = CompleteTaxonomyModel(
            vocab_size=self.tokenizer.vocab_size,
            num_classes_per_rank=self.num_classes_per_rank,
        )

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.info(f"Unexpected keys (ignored): {unexpected}")

        self.model.to(self.device)
        self.model.eval()

        # ── Ecosystem analyzer ───────────────────────────────────────────
        self.analyzer = EcosystemAnalyzer(embedding_dim=256).to(self.device)
        self.analyzer.eval()

        self._loaded = True
        logger.info(
            f"Model loaded | device={self.device} | "
            f"checkpoint={ckpt_path.name} | "
            f"classes_per_rank={self.num_classes_per_rank} | "
            f"rank_thresholds={self.rank_thresholds} | "
            f"min_classification_rank={self.min_classification_rank} "
            f"(specificity>={_MIN_SPECIFICITY})"
        )

    def _tokenize_batch(self, sequences: List[str]) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(s) for s in sequences]
        return torch.tensor(tokens, dtype=torch.long).to(self.device)

    def predict_sequence(self, sequence: str, seq_id: str = 'query') -> Dict:
        """Classify a single DNA sequence."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        with torch.no_grad():
            ids = self._tokenize_batch([sequence])
            out = self.model(ids)
            logits_single = {r: out['logits'][r] for r in RANKS}
            fb = taxonomic_fallback(
                logits_single, self.id_to_label, self.threshold)

        return {
            'id': seq_id,
            'sequence_length': len(sequence),
            'path': 'classification' if fb['specific_enough'] else 'novel_candidate',
            'reported_rank': fb['reported_rank'],
            'reported_taxon': fb['reported_taxon'],
            'reported_confidence': fb['reported_confidence'],
            'fallback_occurred': fb['fallback_occurred'],
            'specific_enough': fb['specific_enough'],
            'taxonomy': fb['all_ranks'],
        }

    def predict_sample(self, sequences: List[str],
                       seq_ids: Optional[List[str]] = None) -> Dict:
        """
        Full pipeline on a batch of sequences from one eDNA sample.

        Routing logic:
          - specific_enough=True  → classification results
            (reported rank >= MIN_CLASSIFICATION_RANK, default: order)
          - specific_enough=False → novel candidate → HDBSCAN clustering
            (reported rank is domain/kingdom/phylum/class — too coarse)

        This ensures Domain/Kingdom always-confident sequences don't swallow
        the novel detection path.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if seq_ids is None:
            seq_ids = [f'seq_{i}' for i in range(len(sequences))]

        # ── Forward pass in batches ──────────────────────────────────────
        all_logits = {r: [] for r in RANKS}
        all_projected = []

        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch_seqs = sequences[i:i + self.batch_size]
                ids = self._tokenize_batch(batch_seqs)
                out = self.model(ids)
                for r in RANKS:
                    all_logits[r].append(out['logits'][r].cpu())
                all_projected.append(out['projected'].cpu())

        all_logits = {r: torch.cat(all_logits[r], dim=0) for r in RANKS}
        all_projected = torch.cat(all_projected, dim=0)

        # ── Confidence routing ───────────────────────────────────────────
        classified_results = []
        novel_seqs = []
        novel_ids = []
        novel_projected = []
        novel_indices = []

        for i, (seq, sid) in enumerate(zip(sequences, seq_ids)):
            logits_i = {r: all_logits[r][i:i + 1] for r in RANKS}
            fb = taxonomic_fallback(
                logits_i, self.id_to_label, self.threshold)

            if fb['specific_enough']:
                # Confident at Order or Family level — classify normally
                classified_results.append({
                    'id': sid,
                    'sequence_length': len(seq),
                    'path': 'classification',
                    'reported_rank': fb['reported_rank'],
                    'reported_taxon': fb['reported_taxon'],
                    'reported_confidence': fb['reported_confidence'],
                    'fallback_occurred': fb['fallback_occurred'],
                    'taxonomy': fb['all_ranks'],
                })
            else:
                # Only confident at Domain/Kingdom/Phylum/Class — novel candidate
                novel_seqs.append(seq)
                novel_ids.append(sid)
                novel_projected.append(all_projected[i])
                novel_indices.append(i)
                logger.debug(
                    f"{sid} → novel candidate "
                    f"(best rank: {fb['reported_rank']}, "
                    f"specificity: {fb['reported_specificity']} < {_MIN_SPECIFICITY})"
                )

        logger.info(
            f"Routing: {len(classified_results)} classified, "
            f"{len(novel_seqs)} novel candidates "
            f"(min_rank={self.min_classification_rank})"
        )

        # ── Novel taxa clustering ────────────────────────────────────────
        novel_report = None
        novel_embeddings_for_history = None
        if len(novel_seqs) >= self.hdbscan_min_cluster:
            stacked = torch.stack(novel_projected)
            novel_report = self._cluster_novel(
                novel_seqs, novel_ids, stacked,
                {r: all_logits[r][novel_indices] for r in RANKS},
            )
            novel_embeddings_for_history = stacked.numpy().tolist()
        elif novel_seqs:
            # Not enough for HDBSCAN — report them as unclustered novel candidates
            logger.info(
                f"{len(novel_seqs)} novel candidates below HDBSCAN minimum "
                f"({self.hdbscan_min_cluster}) — reported as unclustered"
            )
            novel_report = {
                'n_clusters': 0,
                'noise_count': len(novel_seqs),
                'noise_fraction': 1.0,
                'noise_rank_dist': {},
                'clusters': {},
                'unclustered_ids': novel_ids,
                'note': (
                    f'Too few novel candidates ({len(novel_seqs)}) for '
                    f'HDBSCAN (min={self.hdbscan_min_cluster}). '
                    f'Listed as unclustered novel candidates.'
                ),
            }

        # ── Ecosystem analysis ───────────────────────────────────────────
        ecosystem = None
        if len(sequences) >= 2:
            proj_dev = all_projected.to(self.device)
            logits_dev = {r: all_logits[r].to(self.device) for r in RANKS}
            with torch.no_grad():
                ecosystem = self.analyzer.analyze(
                    proj_dev, logits_dev, self.id_to_label)

        return {
            'summary': {
                'total_sequences': len(sequences),
                'classified': len(classified_results),
                'novel_candidates': len(novel_seqs),
                'novel_clusters': (novel_report['n_clusters']
                                   if novel_report else 0),
                'confidence_threshold': self.threshold,
                'rank_thresholds': self.rank_thresholds,
                'min_classification_rank': self.min_classification_rank,
            },
            'classified': classified_results,
            'novel_taxa': novel_report,
            'ecosystem': ecosystem,
            '_novel_embeddings': novel_embeddings_for_history,
            '_novel_ids': novel_ids if novel_seqs else [],
        }

    def _cluster_novel(self, sequences: List[str], seq_ids: List[str],
                       projected: torch.Tensor, logits: Dict) -> Dict:
        """Run HDBSCAN on low-confidence / low-specificity embeddings."""
        try:
            import hdbscan as _hdbscan
        except ImportError:
            logger.warning("hdbscan not installed — skipping clustering")
            return None

        emb_np = projected.numpy().astype(np.float32)
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster,
            min_samples=self.hdbscan_min_samples,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(emb_np).tolist()

        unique_clusters = sorted(set(labels) - {-1})
        n_clusters = len(unique_clusters)
        noise_count = labels.count(-1)

        cluster_report = {}
        for c in unique_clusters:
            member_idx = [i for i, l in enumerate(labels) if l == c]
            member_ids = [seq_ids[i] for i in member_idx]

            # Taxonomic fallback vote across cluster members (average logits)
            logits_c = {r: logits[r][member_idx] for r in RANKS}
            avg_logits = {r: logits_c[r].mean(dim=0, keepdim=True) for r in RANKS}
            fb = taxonomic_fallback(
                avg_logits, self.id_to_label, self.threshold)

            assessment = (
                f"Novel candidate — classifier resolves to "
                f"{fb['reported_rank']}: {fb['reported_taxon']} "
                f"(conf={fb['reported_confidence']:.2f}, "
                f"specificity={fb['reported_specificity']} < {_MIN_SPECIFICITY})"
            )

            # Novelty score: average cosine distance from nearest known prototype
            cluster_embs = projected[member_idx]
            proto = self.model.prototypes.data.cpu()
            cos_sim = F.cosine_similarity(
                cluster_embs.unsqueeze(1), proto.unsqueeze(0), dim=2)
            novelty_score = round(
                1.0 - cos_sim.max(dim=1).values.mean().item(), 4)

            cluster_report[f'cluster_{c}'] = {
                'size': len(member_idx),
                'member_ids': member_ids,
                'reported_rank': fb['reported_rank'],
                'reported_taxon': fb['reported_taxon'],
                'reported_confidence': fb['reported_confidence'],
                'fallback_occurred': fb['fallback_occurred'],
                'taxonomy_profile': fb['all_ranks'],
                'assessment': assessment,
                'novelty_score': novelty_score,
            }

        # Noise point summary
        noise_idx = [i for i, l in enumerate(labels) if l == -1]
        noise_rank_dist = {}
        for i in noise_idx:
            logits_i = {r: logits[r][i:i + 1] for r in RANKS}
            fb = taxonomic_fallback(
                logits_i, self.id_to_label, self.threshold)
            rr = fb['reported_rank']
            noise_rank_dist[rr] = noise_rank_dist.get(rr, 0) + 1

        return {
            'n_clusters': n_clusters,
            'noise_count': noise_count,
            'noise_fraction': round(noise_count / max(len(labels), 1), 4),
            'noise_rank_dist': noise_rank_dist,
            'clusters': cluster_report,
            'note': (
                'Sequences here were confident only at Domain/Kingdom/Phylum/Class '
                'level — below the minimum specificity for classification. '
                'Noise points are the strongest novel taxa candidates.'
            ),
        }

    def get_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Get 256-dim projected embeddings for a batch of sequences."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        all_proj = []
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = sequences[i:i + self.batch_size]
                ids = self._tokenize_batch(batch)
                out = self.model(ids)
                all_proj.append(out['projected'].cpu())
        return torch.cat(all_proj, dim=0).numpy()