"""Statistical tests used by the dashboard and anomaly checks.

Functions are permissive about SciPy availability: when SciPy is present
we return full test statistics and p-values; otherwise we return basic
summary statistics and None for p-values.
"""
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from scipy.stats import wilcoxon, ttest_rel, ks_2samp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def paired_statistics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Return basic paired summary stats: mean_diff, std_diff, n."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape for paired tests")
    diff = a - b
    return {"mean_diff": float(np.nanmean(diff)),
            "std_diff": float(np.nanstd(diff, ddof=1)) if diff.size > 1 else 0.0,
            "n": int(np.count_nonzero(~np.isnan(diff)))}


def run_paired_tests(a, b) -> Dict[str, Optional[Tuple[float, float]]]:
    """Run paired tests on arrays `a` and `b`.

    Returns a dictionary with keys:
      - 'wilcoxon': (stat, p) or None
      - 'ttest_rel': (stat, p) or None
      - 'ks_2samp': (stat, p) or None
      - 'summary': basic summary from `paired_statistics`

    If SciPy is not available, test entries are None and only 'summary' is provided.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape for paired tests")

    out = {"wilcoxon": None, "ttest_rel": None, "ks_2samp": None}
    out["summary"] = paired_statistics(a, b)

    if SCIPY_AVAILABLE:
        # remove NaNs pairwise
        mask = ~np.isnan(a) & ~np.isnan(b)
        a2 = a[mask]
        b2 = b[mask]
        if a2.size == 0:
            return out
        try:
            if a2.size > 0:
                w_stat, w_p = wilcoxon(a2, b2, zero_method='wilcox', alternative='two-sided')
                out['wilcoxon'] = (float(w_stat), float(w_p))
        except Exception:
            out['wilcoxon'] = None
        try:
            t_stat, t_p = ttest_rel(a2, b2, nan_policy='omit')
            out['ttest_rel'] = (float(t_stat), float(t_p))
        except Exception:
            out['ttest_rel'] = None
        try:
            ks_stat, ks_p = ks_2samp(a2, b2)
            out['ks_2samp'] = (float(ks_stat), float(ks_p))
        except Exception:
            out['ks_2samp'] = None

    return out


def cohen_d(a, b) -> float:
    """Compute Cohen's d for paired samples (mean_diff / sd_diff).

    Returns 0.0 if inputs are invalid or sd == 0.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape for effect size")
    diff = a - b
    mean_diff = np.nanmean(diff)
    sd = np.nanstd(diff, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(mean_diff / sd)
