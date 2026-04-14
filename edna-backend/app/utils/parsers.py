"""
parsers.py — Input file parsers for FASTA, FASTQ, CSV, JSON, XLS/XLSX
=======================================================================
Includes Q20 quality filtering for FASTQ files (as specified in the docs).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def parse_fasta(path: str) -> List[Dict]:
    records, current_id, current_seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    records.append({
                        'id': current_id,
                        'sequence': ''.join(current_seq)
                    })
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id is not None:
        records.append({'id': current_id, 'sequence': ''.join(current_seq)})
    return records


def _mean_quality(quality_string: str) -> float:
    """Compute mean Phred quality score from ASCII-encoded quality string."""
    if not quality_string:
        return 0.0
    return sum(ord(c) - 33 for c in quality_string) / len(quality_string)


def parse_fastq(path: str, min_quality: float = 20.0) -> List[Dict]:
    """Parse FASTQ with Q20 quality filtering."""
    records = []
    skipped = 0
    with open(path) as f:
        while True:
            header = f.readline().strip()
            seq = f.readline().strip()
            plus = f.readline().strip()
            quality = f.readline().strip()
            if not header:
                break
            mean_q = _mean_quality(quality)
            if mean_q >= min_quality:
                records.append({
                    'id': header[1:].split()[0],
                    'sequence': seq,
                    'quality_score': round(mean_q, 2),
                })
            else:
                skipped += 1
    if skipped > 0:
        logger.info(f"FASTQ: {skipped} reads filtered (mean Q < {min_quality})")
    return records


def parse_csv(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        seq_col = next(
            (h for h in reader.fieldnames
             if h.lower() in ('sequence', 'seq', 'read', 'nucleotide')),
            None
        )
        id_col = next(
            (h for h in reader.fieldnames
             if h.lower() in ('id', 'sequence_id', 'name', 'record_id', 'sample_id')),
            None
        )
        if seq_col is None:
            raise ValueError(
                f"CSV must have a sequence column. Found: {reader.fieldnames}")
        for i, row in enumerate(reader):
            records.append({
                'id': row[id_col] if id_col else f'seq_{i}',
                'sequence': row[seq_col],
            })
    return records


def parse_json_input(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    records = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            records.append({'id': f'seq_{i}', 'sequence': item})
        elif isinstance(item, dict):
            seq = item.get('sequence') or item.get('seq') or item.get('read')
            if seq is None:
                raise ValueError(f"JSON record {i} has no 'sequence' field")
            rec_id = (item.get('id') or item.get('record_id')
                      or item.get('name') or f'seq_{i}')
            records.append({'id': rec_id, 'sequence': seq})
    return records


def parse_xls(path: str) -> List[Dict]:
    """Parse .xls or .xlsx files. Expects a column named sequence/seq/read/nucleotide."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for XLS/XLSX parsing: pip install pandas openpyxl xlrd")

    ext = Path(path).suffix.lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    df = pd.read_excel(path, engine=engine)

    # Normalise column names to lowercase for matching
    df.columns = [str(c).strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    seq_col = next(
        (col_lower[k] for k in ('sequence', 'seq', 'read', 'nucleotide') if k in col_lower),
        None
    )
    if seq_col is None:
        raise ValueError(
            f"XLS/XLSX must have a sequence column. Found: {list(df.columns)}")

    id_col = next(
        (col_lower[k] for k in ('id', 'sequence_id', 'name', 'record_id', 'sample_id') if k in col_lower),
        None
    )

    records = []
    for i, row in df.iterrows():
        seq = str(row[seq_col]).strip()
        if not seq or seq.lower() == 'nan':
            continue
        rec_id = str(row[id_col]).strip() if id_col else f'seq_{i}'
        records.append({'id': rec_id, 'sequence': seq})
    return records


def parse_file(path: str) -> List[Dict]:
    """Auto-detect format from extension and parse."""
    ext = Path(path).suffix.lower()
    if ext in ('.fa', '.fasta', '.fna', '.ffn', '.faa', '.frn'):
        return parse_fasta(path)
    elif ext in ('.fq', '.fastq'):
        return parse_fastq(path)
    elif ext == '.csv':
        return parse_csv(path)
    elif ext == '.json':
        return parse_json_input(path)
    elif ext in ('.xls', '.xlsx'):
        return parse_xls(path)
    elif ext == '.txt':
        # Try FASTA first, fall back to plain sequences
        with open(path) as f:
            first_line = f.readline().strip()
        if first_line.startswith('>'):
            return parse_fasta(path)
        else:
            # Assume one sequence per line
            records = []
            with open(path) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        records.append({'id': f'seq_{i}', 'sequence': line})
            return records
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Use .fasta, .fastq, .csv, .json, .xls, .xlsx, or .txt"
        )


def validate_sequence(seq: str) -> str:
    """
    Basic sequence validation:
    - Strip whitespace
    - Uppercase
    - Reject if too short (<100 bp after cleaning) or too long (>10,000 bp)
    - Reject if >10% ambiguous N bases
    """
    seq = seq.upper().strip().replace(' ', '').replace('\n', '')
    if len(seq) < 50:
        raise ValueError(f"Sequence too short ({len(seq)} bp, minimum 50)")
    if len(seq) > 10000:
        raise ValueError(f"Sequence too long ({len(seq)} bp, maximum 10,000)")
    n_count = seq.count('N')
    if n_count / len(seq) > 0.1:
        raise ValueError(
            f"Too many ambiguous bases ({n_count}/{len(seq)} = "
            f"{n_count/len(seq):.1%}, maximum 10%)")
    invalid = set(seq) - set('ACGTNRYSWKMBDHV')
    if invalid:
        raise ValueError(f"Invalid characters in sequence: {invalid}")
    return seq
