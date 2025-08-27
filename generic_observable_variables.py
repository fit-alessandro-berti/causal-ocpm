from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple
import math
from itertools import combinations

import numpy as np
import pandas as pd


# ---------- Internal utilities ----------

def _ensure_set(x: Iterable[str]) -> Set[str]:
    return set(x) if not isinstance(x, set) else x


def _event_to_oids(ocel) -> Dict[str, Set[str]]:
    """
    Map each event id to the set of object ids linked to it via ocel.relations.
    """
    rel = ocel.relations[['ocel:eid', 'ocel:oid']].dropna()
    # Groupby returns Series: eid -> set(oids)
    return rel.groupby('ocel:eid')['ocel:oid'].apply(set).to_dict()


def _object_to_eids(ocel) -> Dict[str, Set[str]]:
    """
    Map each object id to the set of event ids it participates in.
    """
    rel = ocel.relations[['ocel:eid', 'ocel:oid']].dropna()
    return rel.groupby('ocel:oid')['ocel:eid'].apply(set).to_dict()


def _execution_event_ids(ocel, obj_ids: Iterable[str], ev_to_oids: Optional[Mapping[str, Set[str]]] = None) -> Set[str]:
    """
    Compute E' = {e : O(e) ⊆ O'} using the subset/closure definition.
    """
    objs = _ensure_set(obj_ids)
    if ev_to_oids is None:
        ev_to_oids = _event_to_oids(ocel)
    kept = {eid for eid, oids in ev_to_oids.items() if oids.issubset(objs)}
    return kept


def _execution_event_frame(ocel, eids: Set[str]) -> pd.DataFrame:
    """
    Events dataframe restricted to E', sorted by timestamp (UTC).
    """
    if len(eids) == 0:
        return ocel.events.iloc[0:0].copy()
    ev = ocel.events[ocel.events['ocel:eid'].isin(eids)].copy()
    # Ensure timezone-aware UTC
    ts = ev['ocel:timestamp']
    if hasattr(ts.dt, "tz_localize"):
        if ts.dt.tz is None:
            ev['ocel:timestamp'] = ts.dt.tz_localize('UTC')
        else:
            ev['ocel:timestamp'] = ts.dt.tz_convert('UTC')
    ev.sort_values('ocel:timestamp', inplace=True)
    return ev


def _execution_relations_frame(ocel, eids: Set[str], obj_ids: Set[str]) -> pd.DataFrame:
    """
    Relations restricted to E' and O'.
    """
    if len(eids) == 0 or len(obj_ids) == 0:
        return ocel.relations.iloc[0:0][['ocel:eid', 'ocel:oid', 'ocel:timestamp']].copy()
    rel = ocel.relations[['ocel:eid', 'ocel:oid', 'ocel:timestamp']]
    rel = rel[rel['ocel:eid'].isin(eids) & rel['ocel:oid'].isin(obj_ids)].copy()
    # Normalize timestamps to UTC for lifespan computations
    ts = rel['ocel:timestamp']
    if hasattr(ts.dt, "tz_localize"):
        if ts.dt.tz is None:
            rel['ocel:timestamp'] = ts.dt.tz_localize('UTC')
        else:
            rel['ocel:timestamp'] = ts.dt.tz_convert('UTC')
    return rel


def _safe_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def _gini_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(1.0 - np.sum(p * p))


# ---------- Indicator functions (each uses closure E' from O') ----------

def count_events(ocel, obj_ids: Iterable[str]) -> int:
    """N_E: number of events in E'."""
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, obj_ids, ev_to_oids)
    return int(len(eids))


def count_objects(obj_ids: Iterable[str]) -> int:
    """N_O: number of objects in O'."""
    return int(len(_ensure_set(obj_ids)))


def distinct_activity_count(ocel, obj_ids: Iterable[str]) -> int:
    """U_A: number of distinct activities in E'."""
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, obj_ids, ev_to_oids)
    ev = _execution_event_frame(ocel, eids)
    return int(ev['ocel:activity'].nunique())


def activity_entropy(ocel, obj_ids: Iterable[str]) -> float:
    """H_act: Shannon entropy of activity distribution in E' (nats)."""
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, obj_ids, ev_to_oids)
    ev = _execution_event_frame(ocel, eids)
    if ev.empty:
        return 0.0
    counts = ev['ocel:activity'].value_counts().to_numpy()
    return _safe_entropy(counts)


def activity_gini(ocel, obj_ids: Iterable[str]) -> float:
    """G_act: 1 - sum_a p_a^2 over activities in E'."""
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, obj_ids, ev_to_oids)
    ev = _execution_event_frame(ocel, eids)
    if ev.empty:
        return 0.0
    counts = ev['ocel:activity'].value_counts().to_numpy()
    return _gini_from_counts(counts)


def average_object_degree(ocel, obj_ids: Iterable[str]) -> float:
    """
    Average number of events per object within the execution: mean_o |{e ∈ E' : o ∈ O(e)}|.
    """
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0.0
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, objs, ev_to_oids)
    rel = _execution_relations_frame(ocel, eids, objs)
    if rel.empty:
        return 0.0
    deg = rel.groupby('ocel:oid')['ocel:eid'].nunique()
    # Include objects that happen to have 0 degree (if any provided)
    if len(deg) < len(objs):
        zeros = pd.Series(0, index=sorted(list(objs - set(deg.index))))
        deg = pd.concat([deg, zeros])
    return float(deg.mean())


def max_object_degree(ocel, obj_ids: Iterable[str]) -> int:
    """Maximum number of events any object participates in within E'."""
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, objs, ev_to_oids)
    rel = _execution_relations_frame(ocel, eids, objs)
    if rel.empty:
        return 0
    deg = rel.groupby('ocel:oid')['ocel:eid'].nunique()
    if deg.empty:
        return 0
    return int(deg.max())


def average_event_coobject_cardinality(ocel, obj_ids: Iterable[str]) -> float:
    """
    Average |O(e)| over events e ∈ E' (each event's number of related objects).
    """
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0.0
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, objs, ev_to_oids)
    if not eids:
        return 0.0
    sizes = [len(ev_to_oids[eid]) for eid in eids]
    return float(np.mean(sizes)) if sizes else 0.0


def max_event_coobject_cardinality(ocel, obj_ids: Iterable[str]) -> int:
    """Maximum |O(e)| over events e ∈ E'."""
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, objs, ev_to_oids)
    if not eids:
        return 0
    sizes = [len(ev_to_oids[eid]) for eid in eids]
    return int(max(sizes)) if sizes else 0


def makespan_seconds(ocel, obj_ids: Iterable[str]) -> float:
    """
    T_span (seconds): max(timestamp) - min(timestamp) over E'.
    """
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, obj_ids, ev_to_oids)
    ev = _execution_event_frame(ocel, eids)
    if ev.empty:
        return 0.0
    tmin = ev['ocel:timestamp'].min()
    tmax = ev['ocel:timestamp'].max()
    return float((tmax - tmin).total_seconds())


def event_rate_per_second(ocel, obj_ids: Iterable[str], epsilon_seconds: float = 1.0) -> float:
    """
    λ_evt := N_E / max(T_span, ε), measured in events per second.
    Use epsilon_seconds to avoid division by zero for zero-span executions.
    """
    NE = count_events(ocel, obj_ids)
    span = makespan_seconds(ocel, obj_ids)
    denom = max(span, float(epsilon_seconds))
    return float(NE / denom) if denom > 0 else 0.0


def mean_inter_event_time_seconds(ocel, obj_ids: Iterable[str]) -> float:
    """
    Mean time difference (seconds) between consecutive events in E' (sorted by timestamp).
    Returns 0.0 if fewer than 2 events.
    """
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, obj_ids, ev_to_oids)
    ev = _execution_event_frame(ocel, eids)
    if len(ev) < 2:
        return 0.0
    diffs = ev['ocel:timestamp'].diff().dropna().dt.total_seconds()
    return float(diffs.mean()) if not diffs.empty else 0.0


def average_object_lifespan_seconds(ocel, obj_ids: Iterable[str]) -> float:
    """
    For each object o ∈ O', take:
        L(o) = max_{e ∈ E' with o in O(e)} t(e) - min_{e ∈ E' with o in O(e)} t(e)
    Return the average L(o) over objects with at least one event in E'.
    If no object participates in E', return 0.0.
    """
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0.0
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, objs, ev_to_oids)
    rel = _execution_relations_frame(ocel, eids, objs)
    if rel.empty:
        return 0.0
    # Lifespan per object
    grp = rel.groupby('ocel:oid')['ocel:timestamp']
    spans = (grp.max() - grp.min()).dt.total_seconds()
    if spans.empty:
        return 0.0
    return float(spans.mean())


def interaction_density(ocel, obj_ids: Iterable[str]) -> float:
    """
    ρ_int := |I'| / max({|O'| choose 2}, 1), where
    I' = { {o,o'} ⊆ O' : ∃ e ∈ E' with {o,o'} ⊆ O(e) } (undirected).
    """
    objs = _ensure_set(obj_ids)
    n = len(objs)
    if n < 2:
        return 0.0
    ev_to_oids = _event_to_oids(ocel)
    eids = _execution_event_ids(ocel, objs, ev_to_oids)
    if not eids:
        return 0.0
    edges = set()
    for eid in eids:
        oset = ev_to_oids[eid]
        if len(oset) >= 2:
            for u, v in combinations(sorted(oset), 2):
                if u in objs and v in objs:
                    edges.add((u, v))
    denom = n * (n - 1) / 2
    return float(len(edges) / denom) if denom > 0 else 0.0


def type_diversity_count(ocel, obj_ids: Iterable[str]) -> int:
    """
    U_T: number of distinct object types present in O'.
    """
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0
    obj_df = ocel.objects[['ocel:oid', 'ocel:type']]
    types = obj_df[obj_df['ocel:oid'].isin(objs)]['ocel:type'].dropna().unique()
    return int(len(types))


def type_entropy(ocel, obj_ids: Iterable[str]) -> float:
    """
    H_type: Shannon entropy (nats) of the distribution of object types in O'.
    """
    objs = _ensure_set(obj_ids)
    if not objs:
        return 0.0
    obj_df = ocel.objects[['ocel:oid', 'ocel:type']]
    sub = obj_df[obj_df['ocel:oid'].isin(objs)]
    if sub.empty:
        return 0.0
    counts = sub['ocel:type'].value_counts().to_numpy()
    return _safe_entropy(counts)


# ---------- Aggregator ----------

def compute_all_indicators(ocel, obj_ids: Iterable[str]) -> Dict[str, float]:
    """
    Compute a battery of generic indicators on the execution induced by obj_ids.

    Returns a dict with keys:
        N_E, N_O, U_A, H_act, G_act,
        T_span_s, lambda_evt_per_s, mean_inter_event_s,
        avg_obj_degree, max_obj_degree,
        avg_event_coobj, max_event_coobj,
        avg_object_lifespan_s,
        interaction_density,
        U_T, H_type
    """
    objs = _ensure_set(obj_ids)

    indicators = {
        'N_E': count_events(ocel, objs),
        'N_O': count_objects(objs),
        'U_A': distinct_activity_count(ocel, objs),
        'H_act': activity_entropy(ocel, objs),
        'G_act': activity_gini(ocel, objs),
        'T_span_s': makespan_seconds(ocel, objs),
        'lambda_evt_per_s': event_rate_per_second(ocel, objs),
        'mean_inter_event_s': mean_inter_event_time_seconds(ocel, objs),
        'avg_obj_degree': average_object_degree(ocel, objs),
        'max_obj_degree': max_object_degree(ocel, objs),
        'avg_event_coobj': average_event_coobject_cardinality(ocel, objs),
        'max_event_coobj': max_event_coobject_cardinality(ocel, objs),
        'avg_object_lifespan_s': average_object_lifespan_seconds(ocel, objs),
        'interaction_density': interaction_density(ocel, objs),
        'U_T': type_diversity_count(ocel, objs),
        'H_type': type_entropy(ocel, objs),
    }
    return indicators


if __name__ == "__main__":
    import pm4py
    ocel = pm4py.read_ocel("C:/example_log.jsonocel")
    print(ocel)
    print(ocel.objects["ocel:oid"].unique())

    print(compute_all_indicators(ocel, {"d1", "i1", "i2", "i3", "o1"}))
