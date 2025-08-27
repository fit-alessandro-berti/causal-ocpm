from __future__ import annotations

from typing import Iterable, Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd
from semopy import Model
import pm4py

# Import your indicator function from the previous file
from generic_observable_variables import compute_all_indicators


# ----------------------------- Data assembly ----------------------------- #

def build_execution_indicator_table(
    ocel,
    object_executions: Iterable[Iterable[str]],
    drop_zero_event_executions: bool = True
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per process execution and one column per indicator.
    `object_executions` is an iterable of sets/lists of object ids (strings).
    """
    rows: List[Dict[str, Any]] = []
    for i, obj_set in enumerate(object_executions):
        metrics = compute_all_indicators(ocel, set(obj_set))
        metrics["exec_id"] = i
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("exec_id").sort_index()

    if drop_zero_event_executions:
        df = df[df["N_E"] > 0].copy()

    return df


# ----------------------------- Pre-processing ---------------------------- #

def _is_finite_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and np.isfinite(s.to_numpy(dtype="float64")).all()

def prepare_sem_dataframe(
    df_exec: pd.DataFrame,
    target_cols: List[str],
    var_tol: float = 1e-12,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Subset, clean, and standardize the execution indicator table.

    Steps:
      - keep only target columns that exist in df_exec;
      - drop rows with any NaN/Inf in those columns;
      - drop columns with (near) zero variance;
      - z-score standardize remaining columns.

    Returns the cleaned/stats-stable DataFrame and the list of kept columns.
    """
    present = [c for c in target_cols if c in df_exec.columns]
    if not present:
        raise ValueError("None of the requested SEM columns are present in df_exec.")

    X = df_exec[present].copy()

    # Coerce to float and drop non-finite
    for c in present:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # Drop near-constant columns (zero variance -> singular covariance)
    std = X.std(ddof=0)
    keep = std[std > var_tol].index.tolist()
    X = X[keep]

    # Standardize (z-scores) to improve conditioning
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    # After standardization, drop any columns that became NaN (possible if std=0)
    X = X.dropna(axis=1, how="any")

    kept_cols = X.columns.tolist()
    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError(
            f"Not enough clean data for SEM after preprocessing: rows={X.shape[0]} cols={X.shape[1]}"
        )

    return X, kept_cols


# ----------------------------- SEM construction -------------------------- #

def build_model_description(
    kept_cols: List[str],
    w_inds_pref: List[str],
    cx_inds_pref: List[str],
    outcome: str,
    structural_controls: List[str],
) -> Tuple[str, List[str]]:
    """
    Build a lavaan-style model string given available columns.
    If a latent has < 2 available indicators, downgrade it to an observed regressor.
    Returns (model_desc, predictors_in_structural).
    """
    # Determine available indicators for each latent
    w_inds = [c for c in w_inds_pref if c in kept_cols]
    cx_inds = [c for c in cx_inds_pref if c in kept_cols]

    # We avoid strong collinearity by NOT using highly redundant pairs.
    # For workload W, we deliberately exclude 'inv_mean_inter_event_s' (near-duplicate of rate).
    # Choose compact sets:
    #   W  =~ lambda_evt_per_s + N_O
    #   Cx =~ H_act + U_A + interaction_density + avg_event_coobj + H_type
    w_latent = len(w_inds) >= 2
    cx_latent = len(cx_inds) >= 2

    lines: List[str] = []
    predictors: List[str] = []

    # Measurement part
    if w_latent:
        lines.append("W  =~ " + " + ".join(w_inds))
        predictors.append("W")
    else:
        # downgrade to observed regressor: prefer lambda_evt_per_s if available, else any present
        w_obs = w_inds[0] if len(w_inds) >= 1 else None
        if w_obs is None:
            # If neither latent nor observed workload indicator is available, skip W entirely
            pass
        else:
            predictors.append(w_obs)

    if cx_latent:
        lines.append("Cx =~ " + " + ".join(cx_inds))
        predictors.append("Cx")
    else:
        cx_obs = cx_inds[0] if len(cx_inds) >= 1 else None
        if cx_obs is None:
            pass
        else:
            predictors.append(cx_obs)

    # Structural part: outcome ~ predictors + controls
    rhs = predictors + [c for c in structural_controls if c in kept_cols]
    rhs_unique = []
    for c in rhs:
        if c not in rhs_unique:
            rhs_unique.append(c)

    if not rhs_unique:
        raise ValueError("No predictors available to regress the outcome on.")

    lines.append(f"{outcome} ~ " + " + ".join(rhs_unique))

    # Optional covariance between latents only if both are present
    if w_latent and cx_latent:
        lines.append("W ~~ Cx")

    desc = "\n".join(lines)
    return desc, rhs_unique


# ----------------------------- Main API ---------------------------------- #

def fit_sem_from_indicators(
    ocel,
    object_executions: Iterable[Iterable[str]],
) -> Dict[str, Any]:
    """
    End-to-end routine:
      - build df_exec from executions;
      - preprocess to a stable numeric table;
      - build a compact SEM;
      - fit with semopy;
      - return model, estimates, factor scores (if latents present), and the cleaned data.
    """
    # 1) Assemble per-execution indicators
    df_exec = build_execution_indicator_table(ocel, object_executions)

    # 2) Choose a compact, non-redundant set of observed variables for SEM
    # Outcome:
    outcome = "T_span_s"

    # Workload indicators (compact; avoid 1/mean_inter_event to reduce collinearity):
    w_inds_pref = ["lambda_evt_per_s", "N_O"]

    # Complexity indicators:
    cx_inds_pref = ["H_act", "U_A", "interaction_density", "avg_event_coobj", "H_type"]

    # Structural controls (always try to include N_O as a direct control)
    structural_controls = ["N_O"]

    # Target columns to request from df_exec
    target_cols = list(set([outcome] + w_inds_pref + cx_inds_pref + structural_controls))

    # 3) Clean and standardize
    X, kept_cols = prepare_sem_dataframe(df_exec, target_cols)

    # 4) Build model description dynamically (may downgrade latents if insufficient indicators)
    model_desc, predictors = build_model_description(
        kept_cols=kept_cols,
        w_inds_pref=w_inds_pref,
        cx_inds_pref=cx_inds_pref,
        outcome=outcome,
        structural_controls=structural_controls,
    )

    # 5) Fit with semopy
    mod = Model(model_desc)
    _ = mod.fit(X)  # semopy uses covariance internally; X is clean and standardized

    # 6) Collect outputs
    result = {
        "model": mod,
        "model_desc": model_desc,
        "clean_data": X,
        "estimates": mod.inspect(),             # parameter table
        "predictors": predictors,
    }

    # Factor scores only if latents are present
    if ("W  ~=" in model_desc) or ("Cx ~=" in model_desc) or ("W  =~" in model_desc) or ("Cx =~" in model_desc):
        try:
            fs = mod.predict_factors(X)
            result["factor_scores"] = fs
        except Exception:
            # If no latents survived or factors cannot be scored, skip silently
            result["factor_scores"] = pd.DataFrame(index=X.index)

    return result


# ----------------------------- Example usage ----------------------------- #
# NOTE: The following is illustrative; adapt as needed in your environment.
#
# from your_ocel_loader import ocel, object_executions
# out = fit_sem_from_indicators(ocel, object_executions)
# print(out["model_desc"])
# print(out["estimates"].head())
# if "factor_scores" in out:
#     print(out["factor_scores"].head())


if __name__ == "__main__":
    ocel = pm4py.read_ocel("C:/recruiting-red.jsonocel")
    applications = ocel.objects[ocel.objects["ocel:type"] == "applications"]["ocel:oid"].unique()
    applications_list = [{applications[i]} for i in range(len(applications)) if i <= 20]
    out = fit_sem_from_indicators(ocel, applications_list)
    print(out["model_desc"])
    print(out["estimates"].head())
    if "factor_scores" in out:
         print(out["factor_scores"].head())
