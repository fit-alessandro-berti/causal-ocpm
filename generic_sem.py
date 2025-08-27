import pandas as pd
import numpy as np
from semopy import Model
from generic_observable_variables import compute_all_indicators
import pm4py


def execute(ocel, object_executions):
    # 1) Assemble one row per execution (no recomputation of indicators beyond calling the provided function)
    rows = []
    for i, obj_set in enumerate(object_executions):
        metrics = compute_all_indicators(ocel, obj_set)  # dict of all generic indicators
        metrics["exec_id"] = i
        # Derived observable: inverse mean inter-event time (large when events are packed)
        eps = 1e-6
        m = metrics.get("mean_inter_event_s", 0.0)
        metrics["inv_mean_inter_event_s"] = 1.0 / max(m, eps)  # safe inverse
        rows.append(metrics)

    df_exec = pd.DataFrame(rows).set_index("exec_id").sort_index()

    # Optional: drop degenerate executions (no events)
    df_exec = df_exec[df_exec["N_E"] > 0].copy()

    # 2) Specify the SEM (measurement + structural)
    model_desc = r"""
    # Measurement (latent =~ observed)
    W  =~ lambda_evt_per_s + N_O + inv_mean_inter_event_s
    Cx =~ H_act + U_A + interaction_density + avg_event_coobj + H_type
    
    # Structural (endogenous ~ predictors)
    T_span_s ~ W + Cx + N_O
    
    # Optional covariance between latent constructs
    W ~~ Cx
    """

    # 3) Fit the SEM
    mod = Model(model_desc)
    _ = mod.fit(df_exec)

    # 4) Inspect parameter estimates (loadings, paths, variances)
    est_table = mod.inspect()          # pandas DataFrame with estimates and statistics
    print(est_table)

    # 5) Obtain per-execution factor scores for the latents (W, Cx)
    factor_scores = mod.predict_factors(df_exec)   # DataFrame with columns ['W', 'Cx']
    print(factor_scores.head())

    # (Optional) Predict/impute model-implied values for observed variables
    # preds = mod.predict(df_exec)  # returns a DataFrame aligned with df_exec

    # (Optional) If you wish to sample synthetic executions:
    # mod_ms = ModelMeans(model_desc)
    # _ = mod_ms.fit(df_exec)
    # sim = generate_data(mod_ms, n=200, drop_lats=True)  # synthetic observed indicators
    # print(sim.head())


if __name__ == "__main__":
    ocel = pm4py.read_ocel("C:/recruiting-red.jsonocel")
    applications = ocel.objects[ocel.objects["ocel:type"] == "applications"]["ocel:oid"].unique()
    applications_list = [{applications[i]} for i in range(len(applications)) if i <= 20]
    print(execute(ocel, applications_list))
