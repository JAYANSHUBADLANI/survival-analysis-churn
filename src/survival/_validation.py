"""Input validation shared across estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def validate_time_event(
    time: object, event: object
) -> tuple[FloatArray, IntArray]:
    """Coerce and validate a right-censored sample ``(time, event)``.

    Parameters
    ----------
    time:
        Follow-up times, one per subject. Must be finite and non-negative.
    event:
        Event indicators: 1 if the event (e.g. churn) was observed at
        ``time``, 0 if the subject was right-censored at ``time``.

    Returns
    -------
    tuple
        ``(time, event)`` as 1-D float and int arrays of equal length.

    Raises
    ------
    ValueError
        If the arrays are empty, mismatched in length, non-finite,
        negative in time, or if ``event`` contains values outside {0, 1}.
    """
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("time and event must be non-empty")
    if t.size != e.size:
        raise ValueError(
            f"time and event have different lengths ({t.size} vs {e.size})"
        )
    if not np.all(np.isfinite(t)):
        raise ValueError("time contains non-finite values")
    if np.any(t < 0):
        raise ValueError("time contains negative values")
    uniq = np.unique(e)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValueError("event must contain only 0 (censored) and 1 (event)")
    return t, e.astype(int)


def validate_covariates(
    X: object, names: list[str] | None
) -> tuple[FloatArray, list[str]]:
    """Coerce a covariate matrix to 2-D float and resolve column names.

    Accepts a :class:`pandas.DataFrame` (column names inferred) or any
    array-like of shape ``(n,)`` or ``(n, p)``.

    Raises
    ------
    ValueError
        If ``X`` contains non-finite values or ``names`` has wrong length.
    """
    if isinstance(X, pd.DataFrame):
        inferred = [str(c) for c in X.columns]
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
        inferred = []
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("X must be 1-D or 2-D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X contains non-finite values")
    p = arr.shape[1]
    if names is not None:
        resolved = [str(c) for c in names]
    elif inferred:
        resolved = inferred
    else:
        resolved = [f"x{j + 1}" for j in range(p)]
    if len(resolved) != p:
        raise ValueError(
            f"expected {p} covariate names, got {len(resolved)}"
        )
    return arr, resolved
