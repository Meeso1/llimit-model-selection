from typing import Any
from collections import defaultdict
import pandas as pd


def compute_value_counts(series: pd.Series) -> defaultdict[str, defaultdict[str, int]]:
    """
    Computes the value counts of a series containing string-keyed dictionaries (possibly nested).
    """
    def process_dict(counts: defaultdict[str, defaultdict[str, int]], d: dict[str, Any], count: int, path: str | None = None):
        for k, v in d.items():
            child_path = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                process_dict(counts, v, count, child_path)
            else:
                counts[child_path][v] += count
    
    counts = defaultdict(lambda: defaultdict(lambda: 0)) # counts[path][value]
    for (d, count) in series.dropna().value_counts().items():
        process_dict(counts, d, count)
        
    return counts