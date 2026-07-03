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


def compute_z_score(value, history, min_samples=3, std_floor=1.0, one_sided=False):
    if len(history) < min_samples:
        return 0.0
    mean = sum(history) / len(history)
    if one_sided and value <= mean:
        return 0.0
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    std = math.sqrt(variance)
    std = max(std, std_floor)
    return abs(value - mean) / std


# -----------------------------------------------------------------------------
# Event Handlers
# -----------------------------------------------------------------------------

def check_data_batch(payload, ctx):
    res = check_response(ctx.tools.batch_profile(payload["batch_id"]))
    if not res:
        return Verdict(alert=False, pillar="checks", reason="tool call failed")

    table = payload.get("table", "unknown")
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
    suspicion = 0.0

    # Hard checks
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

    # Soft checks (boundary proximity)
    rc_mid = (rc_min + rc_max) / 2.0
    rc_hw = (rc_max - rc_min) / 2.0
    if rc_hw > 0:
        rc_dev = abs(row_count - rc_mid) / rc_hw
        if rc_dev > 0.95:
            suspicion += 0.8
        elif rc_dev > 0.85:
            suspicion += 0.4
        elif rc_dev > 0.70:
            suspicion += 0.2

    ma_mid = (ma_min + ma_max) / 2.0
    ma_hw = (ma_max - ma_min) / 2.0
    if ma_hw > 0:
        ma_dev = abs(mean_amount - ma_mid) / ma_hw
        if ma_dev > 0.95:
            suspicion += 0.8
        elif ma_dev > 0.85:
            suspicion += 0.4
        elif ma_dev > 0.70:
            suspicion += 0.2

    if nr_max > 0:
        nr_dev = null_rate_cust / nr_max
        if nr_dev > 0.90:
            suspicion += 0.8
        elif nr_dev > 0.75:
            suspicion += 0.4
        elif nr_dev > 0.60:
            suspicion += 0.2

    if st_max > 0:
        st_dev = staleness_min / st_max
        if st_dev > 0.90:
            suspicion += 0.8
        elif st_dev > 0.75:
            suspicion += 0.4
        elif st_dev > 0.60:
            suspicion += 0.2

    # Rolling statistics checks
    z_rc = compute_z_score(row_count, get_history(ctx, "data_batch_row_count", table), std_floor=10.0)
    z_ma = compute_z_score(mean_amount, get_history(ctx, "data_batch_mean_amount", table), std_floor=1.0)
    z_nr = compute_z_score(null_rate_cust, get_history(ctx, "data_batch_null_rate", table), std_floor=0.002, one_sided=True)
    z_st = compute_z_score(staleness_min, get_history(ctx, "data_batch_staleness", table), std_floor=1.0, one_sided=True)

    for z_name, z_val in [("row_count", z_rc), ("mean_amount", z_ma), ("null_rate", z_nr), ("staleness", z_st)]:
        if z_val > 4.0:
            suspicion += 1.0
            reasons.append(f"Z-score {z_name} violation: {z_val:.2f}")
        elif z_val > 2.5:
            suspicion += 0.4

    if suspicion >= 1.0:
        alert = True
        reasons.append(f"suspicion threshold reached: {suspicion:.2f}")

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
    suspicion = 0.0

    # Hard checks
    if len(violations) > 0:
        alert = True
        reasons.append(f"contract violations: {violations}")
    if freshness_delay_min > fd_max:
        alert = True
        reasons.append(f"freshness_delay violation: {freshness_delay_min}")

    # Soft checks
    if fd_max > 0:
        fd_dev = freshness_delay_min / fd_max
        if fd_dev > 0.90:
            suspicion += 0.8
        elif fd_dev > 0.75:
            suspicion += 0.4
        elif fd_dev > 0.60:
            suspicion += 0.2

    z_fd = compute_z_score(freshness_delay_min, get_history(ctx, "contract_freshness", contract_id), std_floor=1.0, one_sided=True)
    if z_fd > 4.0:
        suspicion += 1.0
        reasons.append(f"Z-score freshness violation: {z_fd:.2f}")
    elif z_fd > 2.5:
        suspicion += 0.4

    if suspicion >= 1.0:
        alert = True
        reasons.append(f"suspicion threshold reached: {suspicion:.2f}")

    if not alert:
        update_history(ctx, "contract_freshness", contract_id, freshness_delay_min)

    return Verdict(alert=alert, pillar="contracts", reason="; ".join(reasons))


def check_lineage_run(payload, ctx):
    # Only depth=1 for optimal cost
    res = check_response(ctx.tools.lineage_graph_slice(payload["run_id"], depth=1))
    if not res:
        return Verdict(alert=False, pillar="lineage", reason="tool call failed")

    job = payload.get("job", "unknown")
    duration_ms = safe_float(res.get("duration_ms"))
    actual_upstream = res.get("actual_upstream") or []
    actual_downstream_count = int(safe_float(res.get("actual_downstream_count")))

    dur_max = get_baseline(ctx, "lineage_duration_ms_max", 5134.9804)

    alert = False
    reasons = []
    suspicion = 0.0

    # Hard checks
    if duration_ms > dur_max:
        alert = True
        reasons.append(f"duration violation: {duration_ms}")
    if len(actual_upstream) == 0:
        alert = True
        reasons.append("actual_upstream is empty")

    # Graph history check
    if "lineage_graph" not in ctx.state:
        ctx.state["lineage_graph"] = {}
    g = ctx.state["lineage_graph"].setdefault(job, {"max_upstream": [], "normal_downstream": 0})

    if g["max_upstream"]:
        missing = set(g["max_upstream"]) - set(actual_upstream)
        if len(missing) > 0:
            alert = True
            reasons.append(f"missing upstream nodes: {list(missing)}")
        if actual_downstream_count < g["normal_downstream"]:
            alert = True
            reasons.append(f"downstream count dropped: {actual_downstream_count} vs normal {g['normal_downstream']}")

    # Soft checks
    if dur_max > 0:
        dur_dev = duration_ms / dur_max
        if dur_dev > 0.90:
            suspicion += 0.8
        elif dur_dev > 0.75:
            suspicion += 0.4
        elif dur_dev > 0.60:
            suspicion += 0.2

    z_dur = compute_z_score(duration_ms, get_history(ctx, "lineage_duration", job), std_floor=300.0, one_sided=True)
    if z_dur > 4.0:
        suspicion += 1.0
        reasons.append(f"Z-score duration violation: {z_dur:.2f}")
    elif z_dur > 2.5:
        suspicion += 0.4

    if suspicion >= 1.0:
        alert = True
        reasons.append(f"suspicion threshold reached: {suspicion:.2f}")

    if not alert:
        update_history(ctx, "lineage_duration", job, duration_ms)
        # Update typical graph structure from clean run
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

    ms_max = get_baseline(ctx, "feature_mean_shift_sigma_max", 0.4095)
    ms_thresh = max(ms_max * 1.3, 0.60)

    alert = False
    reasons = []
    suspicion = 0.0

    # Hard checks
    if abs(mean_shift_sigma) > ms_thresh:
        alert = True
        reasons.append(f"mean_shift_sigma violation: {mean_shift_sigma}")

    # Soft checks (calculated against ms_thresh to prevent false alerts on clean runs)
    if ms_thresh > 0:
        ms_dev = abs(mean_shift_sigma) / ms_thresh
        if ms_dev > 0.90:
            suspicion += 0.8
        elif ms_dev > 0.75:
            suspicion += 0.4
        elif ms_dev > 0.60:
            suspicion += 0.2

    z_ms = compute_z_score(mean_shift_sigma, get_history(ctx, "feature_mean_shift", feature_view), std_floor=0.05)
    z_sm = compute_z_score(serve_mean, get_history(ctx, "feature_serve_mean", feature_view), std_floor=5.0)

    for z_name, z_val in [("mean_shift", z_ms), ("serve_mean", z_sm)]:
        if z_val > 4.0:
            suspicion += 1.0
            reasons.append(f"Z-score {z_name} violation: {z_val:.2f}")
        elif z_val > 2.5:
            suspicion += 0.4

    if suspicion >= 1.0:
        alert = True
        reasons.append(f"suspicion threshold reached: {suspicion:.2f}")

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
    ad_max = get_baseline(ctx, "corpus_avg_doc_age_days_max", 49.7955)

    alert = False
    reasons = []
    suspicion = 0.0

    # Hard checks
    if centroid_shift > cs_max:
        alert = True
        reasons.append(f"centroid_shift violation: {centroid_shift}")
    if avg_doc_age_days > ad_max:
        alert = True
        reasons.append(f"avg_doc_age_days violation: {avg_doc_age_days}")

    # Soft checks
    if cs_max > 0:
        cs_dev = centroid_shift / cs_max
        if cs_dev > 0.90:
            suspicion += 0.8
        elif cs_dev > 0.75:
            suspicion += 0.4
        elif cs_dev > 0.60:
            suspicion += 0.2

    if ad_max > 0:
        ad_dev = avg_doc_age_days / ad_max
        if ad_dev > 0.90:
            suspicion += 0.8
        elif ad_dev > 0.75:
            suspicion += 0.4
        elif ad_dev > 0.60:
            suspicion += 0.2

    z_cs = compute_z_score(centroid_shift, get_history(ctx, "embedding_centroid_shift", corpus), std_floor=0.005, one_sided=True)
    z_ad = compute_z_score(avg_doc_age_days, get_history(ctx, "embedding_doc_age", corpus), std_floor=3.0, one_sided=True)

    for z_name, z_val in [("centroid_shift", z_cs), ("avg_doc_age", z_ad)]:
        if z_val > 4.0:
            suspicion += 1.0
            reasons.append(f"Z-score {z_name} violation: {z_val:.2f}")
        elif z_val > 2.5:
            suspicion += 0.4

    if suspicion >= 1.0:
        alert = True
        reasons.append(f"suspicion threshold reached: {suspicion:.2f}")

    if not alert:
        update_history(ctx, "embedding_centroid_shift", corpus, centroid_shift)
        update_history(ctx, "embedding_doc_age", corpus, avg_doc_age_days)

    return Verdict(alert=alert, pillar="ai_infra", reason="; ".join(reasons))
