import numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

def build_reference(df, raw_fields, n_bins=10):
    reference = {}
    for feature in raw_fields:
        # ... compute edges + expected % for this feature ...
        edges = np.quantile(df[feature], np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(df[feature], bins=edges)
        expected = counts / counts.sum()
        reference[feature] = {"edges": edges.tolist(), "expected": expected.tolist()}
    return reference

def compute_psi(live_df, reference, raw_fields):
    psi_scores = {}
    eps = 1e-4
    for feature in raw_fields:
        edges = np.array(reference[feature]["edges"], dtype=float)
        expected = np.array(reference[feature]["expected"])
        counts, _ = np.histogram(live_df[feature], bins=edges)
        actual = counts / counts.sum()
        eps = 1e-4
        psi = np.sum((actual - expected) * np.log((actual + eps) / (expected + eps)))
        psi_scores[feature] = psi
    return psi_scores

def drift_evidently(reference_df,current_df,raw_fields):
    schema = DataDefinition(numerical_columns=raw_fields)   

    ref_ds = Dataset.from_pandas(reference_df[raw_fields], data_definition=schema)
    cur_ds = Dataset.from_pandas(current_df[raw_fields],   data_definition=schema)

    report  = Report([DataDriftPreset()])
    results = report.run(cur_ds, ref_ds)  
    out = results.dict()
    drift_scores = {}
    for m in out["metrics"]:
        if "ValueDrift" in m["metric_name"]:
            drift_scores[m["config"]["column"]] = float(m["value"])   
    return drift_scores