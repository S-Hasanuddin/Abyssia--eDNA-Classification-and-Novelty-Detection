"""
historical_clusters.py — Track and compare novel clusters across uploads
==========================================================================
Stores novel sequence embeddings from past runs. When new novel sequences
are detected, they are compared against the historical pool to find matches.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from app.config import HISTORY_DB_PATH

logger = logging.getLogger(__name__)


class HistoricalClusterStore:
    """
    Simple JSON-backed store for historical novel cluster embeddings.
    Each entry stores: job_id, timestamp, cluster_id, centroid (256-dim),
    member_count, taxonomy_profile, and novelty_score.
    """

    def __init__(self, db_path: Path = HISTORY_DB_PATH):
        self.db_path = db_path
        self.data = self._load()

    def _load(self) -> Dict:
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Corrupted history DB — starting fresh")
        return {'clusters': [], 'version': 1}

    def _save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_clusters(self, job_id: str, novel_report: Dict,
                     embeddings: Optional[List[List[float]]] = None,
                     novel_ids: Optional[List[str]] = None):
        """
        Store cluster centroids from a completed analysis job.
        """
        if novel_report is None or 'clusters' not in novel_report:
            return

        timestamp = datetime.utcnow().isoformat()
        clusters = novel_report['clusters']

        for cname, cinfo in clusters.items():
            entry = {
                'job_id': job_id,
                'timestamp': timestamp,
                'cluster_id': cname,
                'size': cinfo['size'],
                'reported_rank': cinfo.get('reported_rank', 'unknown'),
                'reported_taxon': cinfo.get('reported_taxon', 'unknown'),
                'reported_confidence': cinfo.get('reported_confidence', 0),
                'novelty_score': cinfo.get('novelty_score', 0),
                'assessment': cinfo.get('assessment', ''),
                'taxonomy_profile': cinfo.get('taxonomy_profile', {}),
            }

            # Compute centroid from member embeddings if available
            if embeddings and novel_ids and cinfo.get('member_ids'):
                member_indices = []
                for mid in cinfo['member_ids']:
                    if mid in novel_ids:
                        idx = novel_ids.index(mid)
                        if idx < len(embeddings):
                            member_indices.append(idx)
                if member_indices:
                    member_embs = np.array(
                        [embeddings[i] for i in member_indices])
                    centroid = member_embs.mean(axis=0).tolist()
                    entry['centroid'] = [round(x, 6) for x in centroid]

            self.data['clusters'].append(entry)

        self._save()
        logger.info(
            f"Stored {len(clusters)} clusters from job {job_id} "
            f"(total history: {len(self.data['clusters'])})")

    def compare_with_history(self, novel_report: Dict,
                             embeddings: Optional[List[List[float]]] = None,
                             novel_ids: Optional[List[str]] = None,
                             similarity_threshold: float = 0.85) -> Dict:
        """
        Compare current novel clusters against historical clusters.
        Returns matches where cosine similarity > threshold.
        """
        if (novel_report is None or not self.data['clusters']
                or embeddings is None):
            return {'matches': [], 'total_historical': len(self.data['clusters'])}

        # Collect historical centroids
        hist_entries = [e for e in self.data['clusters'] if 'centroid' in e]
        if not hist_entries:
            return {'matches': [], 'total_historical': len(self.data['clusters'])}

        hist_centroids = np.array([e['centroid'] for e in hist_entries],
                                  dtype=np.float32)

        # Compute centroids for current clusters
        matches = []
        current_clusters = novel_report.get('clusters', {})

        for cname, cinfo in current_clusters.items():
            if not cinfo.get('member_ids') or not novel_ids:
                continue

            member_indices = []
            for mid in cinfo['member_ids']:
                if mid in novel_ids:
                    idx = novel_ids.index(mid)
                    if idx < len(embeddings):
                        member_indices.append(idx)

            if not member_indices:
                continue

            member_embs = np.array(
                [embeddings[i] for i in member_indices], dtype=np.float32)
            centroid = member_embs.mean(axis=0)

            # Cosine similarity against all historical centroids
            norm_c = centroid / (np.linalg.norm(centroid) + 1e-8)
            norm_h = hist_centroids / (
                np.linalg.norm(hist_centroids, axis=1, keepdims=True) + 1e-8)
            similarities = norm_h @ norm_c

            # Find matches above threshold
            above = np.where(similarities > similarity_threshold)[0]
            for idx in above:
                hist_entry = hist_entries[idx]
                matches.append({
                    'current_cluster': cname,
                    'current_taxon': cinfo.get('reported_taxon', 'unknown'),
                    'current_novelty_score': cinfo.get('novelty_score', 0),
                    'historical_job_id': hist_entry['job_id'],
                    'historical_cluster_id': hist_entry['cluster_id'],
                    'historical_taxon': hist_entry.get('reported_taxon', 'unknown'),
                    'historical_timestamp': hist_entry['timestamp'],
                    'cosine_similarity': round(float(similarities[idx]), 4),
                })

        matches.sort(key=lambda x: x['cosine_similarity'], reverse=True)

        return {
            'matches': matches,
            'total_historical': len(self.data['clusters']),
            'total_current_clusters': len(current_clusters),
            'similarity_threshold': similarity_threshold,
        }

    def get_history_summary(self) -> Dict:
        """Return summary of all historical clusters."""
        clusters = self.data['clusters']
        if not clusters:
            return {'total_entries': 0, 'jobs': []}

        jobs = {}
        for c in clusters:
            jid = c['job_id']
            if jid not in jobs:
                jobs[jid] = {
                    'job_id': jid,
                    'timestamp': c['timestamp'],
                    'cluster_count': 0,
                    'total_sequences': 0,
                }
            jobs[jid]['cluster_count'] += 1
            jobs[jid]['total_sequences'] += c.get('size', 0)

        return {
            'total_entries': len(clusters),
            'jobs': list(jobs.values()),
        }
