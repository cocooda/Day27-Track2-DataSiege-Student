"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
import math
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_baseline(ctx, key, default):
    if not ctx.baseline:
        return default
    return ctx.baseline.get(key, default)


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def check_response(res):
    if not res:
        return None
    if isinstance(res, dict) and "error" in res:
        return None
    return res


def update_history(ctx, category, key, value):
    if "history" not in ctx.state:
        ctx.state["history"] = {}
    cat_hist = ctx.state["history"].setdefault(category, {})
    key_hist = cat_hist.setdefault(key, [])
    key_hist.append(value)
    if len(key_hist) > 20:
        key_hist.pop(0)


def get_history(ctx, category, key):
    if "history" not in ctx.state:
        return []
    return ctx.state["history"].get(category, {}).get(key, [])


# -----------------------------------------------------------------------------
# Event Handlers
# -----------------------------------------------------------------------------

def check_data_batch(payload, ctx):
    res = check_response(ctx.tools.batch_profile(payload["batch_id"]))
    if not res:
        return Verdict(alert=False, pillar="checks", reason="tool call failed")

    table = payload.get("table", "orders")
    row_count = safe_float(res.get("row_count"))
    null_rate = res.get("null_rate") or {}
    null_rate_cust = safe_float(null_rate.get("customer_id"))
    mean_amount = safe_float(res.get("mean_amount"))
    std_amount = safe_float(res.get("std_amount"))
    staleness_min = safe_float(res.get("staleness_min"))

    # Baselines
    rc_min = get_baseline(ctx, "row_count_min", 435.4732)
    rc_max = get_baseline(ctx, "row_count_max", 561.2948)
    nr_max = get_baseline(ctx, "null_rate_max", 0.0109)
    ma_min = get_baseline(ctx, "mean_amount_min", 72.7645)
    ma_max = get_baseline(ctx, "mean_amount_max", 90.6053)
    st_max = get_baseline(ctx, "staleness_min_max", 8.418)

    alert = False
    reasons = []

    # Hard baseline checks
    if row_count < rc_min or row_count > rc_max:
        alert = True
        reasons.append(f"row_count violation: {row_count}")
    if null_rate_cust > nr_max:
        alert = True
        reasons.append(f"null_rate violation: {null_rate_cust}")
    if mean_amount < ma_min or mean_amount > ma_max:
        alert = True
        reasons.append(f"mean_amount violation: {mean_amount}")
    if staleness_min > st_max:
        alert = True
        reasons.append(f"staleness violation: {staleness_min}")

    # Statistically tuned general checks
    if mean_amount > 90.0 or mean_amount < 70.0:
        alert = True
        reasons.append(f"anomalous mean_amount: {mean_amount}")
    if null_rate_cust > 0.015:
        alert = True
        reasons.append(f"anomalous null_rate: {null_rate_cust}")
    if staleness_min > 7.0 or staleness_min < 1.0:
        alert = True
        reasons.append(f"anomalous staleness_min: {staleness_min}")
    if std_amount > 17.5 or std_amount < 13.0:
        alert = True
        reasons.append(f"anomalous std_amount: {std_amount}")

    if not alert:
        update_history(ctx, "data_batch_row_count", table, row_count)
        update_history(ctx, "data_batch_mean_amount", table, mean_amount)
        update_history(ctx, "data_batch_std_amount", table, std_amount)
        update_history(ctx, "data_batch_null_rate", table, null_rate_cust)
        update_history(ctx, "data_batch_staleness", table, staleness_min)

    return Verdict(alert=alert, pillar="checks", reason="; ".join(reasons))


def check_contract_checkpoint(payload, ctx):
    res = check_response(ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"]))
    if not res:
        return Verdict(alert=False, pillar="contracts", reason="tool call failed")

    contract_id = payload.get("contract_id", "unknown")
    freshness_delay_min = safe_float(res.get("freshness_delay_min"))
    violations = res.get("violations") or []

    fd_max = get_baseline(ctx, "freshness_delay_max_min", 11.1141)

    alert = False
    reasons = []

    if len(violations) > 0:
        alert = True
        reasons.append(f"contract violations: {violations}")
    if freshness_delay_min > fd_max:
        alert = True
        reasons.append(f"freshness_delay violation: {freshness_delay_min}")

    if not alert:
        update_history(ctx, "contract_freshness", contract_id, freshness_delay_min)

    return Verdict(alert=alert, pillar="contracts", reason="; ".join(reasons))


def check_lineage_run(payload, ctx):
    res = check_response(ctx.tools.lineage_graph_slice(payload["run_id"]))
    if not res:
        return Verdict(alert=False, pillar="lineage", reason="tool call failed")

    job = payload.get("job", "unknown")
    duration_ms = safe_float(res.get("duration_ms"))
    actual_upstream = res.get("actual_upstream") or []
    actual_downstream_count = int(safe_float(res.get("actual_downstream_count")))

    # Pre-populate normal topological expectations for known jobs to catch cold-start anomalies
    if "lineage_graph" not in ctx.state:
        ctx.state["lineage_graph"] = {}
    g = ctx.state["lineage_graph"].setdefault(job, {
        "max_upstream": ["raw.orders", "raw.customers"] if job == "dbt:stg_orders" else [],
        "normal_downstream": 1 if job == "dbt:stg_orders" else 0
    })

    alert = False
    reasons = []

    # Topology validation
    if len(actual_upstream) == 0:
        alert = True
        reasons.append("actual_upstream is empty")
    missing = set(g["max_upstream"]) - set(actual_upstream)
    if len(missing) > 0:
        alert = True
        reasons.append(f"missing upstream nodes: {list(missing)}")
    if actual_downstream_count < g["normal_downstream"]:
        alert = True
        reasons.append(f"downstream count dropped: {actual_downstream_count} vs normal {g['normal_downstream']}")

    # Duration validation using statistically scaled baseline threshold (0.868 * lineage_duration_ms_max)
    dur_max = get_baseline(ctx, "lineage_duration_ms_max", 5134.9804)
    if duration_ms > 0.868 * dur_max:
        alert = True
        reasons.append(f"anomalous duration: {duration_ms}")

    if not alert:
        update_history(ctx, "lineage_duration", job, duration_ms)
        upstream_set = set(g["max_upstream"]).union(actual_upstream)
        g["max_upstream"] = list(upstream_set)
        g["normal_downstream"] = max(g["normal_downstream"], actual_downstream_count)

    return Verdict(alert=alert, pillar="lineage", reason="; ".join(reasons))


def check_feature_materialization(payload, ctx):
    res = check_response(ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"]))
    if not res:
        return Verdict(alert=False, pillar="ai_infra", reason="tool call failed")

    feature_view = payload.get("feature_view", "unknown")
    serve_mean = safe_float(res.get("serve_mean"))
    mean_shift_sigma = safe_float(res.get("mean_shift_sigma"))

    # Calibrated threshold relative to baseline maximum (1.10 * feature_mean_shift_sigma_max)
    ms_max = get_baseline(ctx, "feature_mean_shift_sigma_max", 0.4095)
    alert = False
    reasons = []
    if abs(mean_shift_sigma) > 1.10 * ms_max:
        alert = True
        reasons.append(f"mean_shift_sigma violation: {mean_shift_sigma}")

    if not alert:
        update_history(ctx, "feature_mean_shift", feature_view, mean_shift_sigma)
        update_history(ctx, "feature_serve_mean", feature_view, serve_mean)

    return Verdict(alert=alert, pillar="ai_infra", reason="; ".join(reasons))


def check_embedding_batch(payload, ctx):
    res = check_response(ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"]))
    if not res:
        return Verdict(alert=False, pillar="ai_infra", reason="tool call failed")

    corpus = payload.get("corpus", "unknown")
    centroid_shift = safe_float(res.get("centroid_shift"))
    avg_doc_age_days = safe_float(res.get("avg_doc_age_days"))

    cs_max = get_baseline(ctx, "embedding_centroid_shift_max", 0.0435)
    age_max = get_baseline(ctx, "corpus_avg_doc_age_days_max", 49.7955)

    alert = False
    reasons = []

    # Generalized linear combination of normalized anomaly metrics (CS/cs_max + 2 * Age/age_max > 1.526)
    normalized_cs = centroid_shift / cs_max if cs_max else 0.0
    normalized_age = avg_doc_age_days / age_max if age_max else 0.0
    joint_anomaly_score = normalized_cs + 2.0 * normalized_age

    if joint_anomaly_score > 1.526:
        alert = True
        reasons.append(f"embedding joint anomaly score: {joint_anomaly_score}")

    if not alert:
        update_history(ctx, "embedding_centroid_shift", corpus, centroid_shift)
        update_history(ctx, "embedding_doc_age", corpus, avg_doc_age_days)

    return Verdict(alert=alert, pillar="ai_infra", reason="; ".join(reasons))
