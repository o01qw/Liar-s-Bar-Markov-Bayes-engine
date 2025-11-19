"""
palomaki_bluff_model.py

Fit a simple logistic model P(bluff | p_feasible) from the Palomäki et al.
PLOS ONE (2016) dataset: final_data_supplementary.sav

You will need:
    pip install pyreadstat scikit-learn pandas
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import LogisticRegression


@dataclass
class LogisticBluffModel:
    """
    Wrapper that turns a fitted scikit-learn LogisticRegression into
    a callable bluff prior function bluff(p_feasible) -> float.
    """
    model: LogisticRegression
    feature_col: str = "p_feasible"

    def __call__(self, p_feasible: float) -> float:
        X = np.array([[p_feasible]], dtype=float)
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)


def fit_logistic_bluff_model(
    sav_path: str | Path,
    *,
    feas_col: str = "p_feasible",
    bluff_col: str = "is_bluff",
) -> Tuple[LogisticBluffModel, pd.DataFrame]:
    """
    Fit P(bluff | p_feasible) from the Palomäki .sav file.

    Parameters
    ----------
    sav_path : str or Path
        Path to final_data_supplementary.sav
    feas_col : str
        Name of the column in your working DataFrame that contains
        the hypergeometric feasibility p_f for that decision.
        If the raw .sav file does not have this column, compute it
        first in a notebook and save back to disk.
    bluff_col : str
        Boolean / {0,1} indicator for whether the decision was a bluff.

    Returns
    -------
    model : LogisticBluffModel
        Callable object f(p_feasible) -> p_bluff
    df : pd.DataFrame
        Cleaned DataFrame actually used for fitting (for inspection).
    """
    sav_path = Path(sav_path)
    df, meta = pyreadstat.read_sav(sav_path)
    print(df.columns)
    # --- EDIT THIS BLOCK ONCE YOU'VE INSPECTED THE COLUMNS ---
    # For example, you might start with something like:
    #   print(df.columns)
    #   df = df.rename(columns={"BLUFF": "is_bluff"})
    # and then compute p_feasible using your card model.
    # ---------------------------------------------------------
    if bluff_col not in df.columns:
        raise KeyError(
            f"Column {bluff_col!r} not found in .sav file. "
            "Open the file once, inspect column names, and either "
            "rename the bluff indicator to {bluff_col!r} or change "
            "the function argument."
        )

    if feas_col not in df.columns:
        raise KeyError(
            f"Column {feas_col!r} not found. "
            "You probably still need to compute hypergeometric "
            "feasibility for each decision and store it in the data."
        )

    work = df[[feas_col, bluff_col]].dropna().copy()
    work[bluff_col] = work[bluff_col].astype(int)

    X = work[[feas_col]].to_numpy(dtype=float)
    y = work[bluff_col].to_numpy(dtype=int)

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
    )
    model.fit(X, y)

    return LogisticBluffModel(model=model, feature_col=feas_col), work
