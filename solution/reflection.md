# Reflection — Data Siege Defense

### Which fault types were hardest to catch, and why?

1. **Subtle distribution shifts and drift (checks & AI infra):** Faults that hover just inside or slightly outside the 3-sigma thresholds (e.g., subtle feature skew or near-threshold embedding centroid shifts) are notoriously difficult. With small sample sizes, establishing a rolling Z-score baseline suffers from **variance starvation**, where normal statistical fluctuations trigger false alarms because the historical variance is artificially tight. We mitigated this by setting standard deviation floors (e.g., `std_floor = 8.0` for serve mean) and using one-sided Z-scores for naturally non-negative metrics.
2. **Lineage graph shape anomalies (lineage):** Detecting missing upstream nodes or orphaned outputs requires dynamic tracking of expected job topologies since payloads do not expose the reference graph. We resolved this by profiling the maximum seen upstream node set and typical downstream counts in `ctx.state` during clean runs.
3. **Expensive AI infra evaluations:** Metrology costs of 2.0 credits for `feature_drift` and `embedding_drift` make comprehensive coverage expensive, but because payloads are identical between clean and faulty states, checking every event remains necessary to prevent missed faults.

### What would you change about your cost/coverage tradeoff, if you had another pass?

With another pass, we would introduce a **dynamic credit allocation manager**:
1. **Adaptive Triage:** Instead of running the metered tool for every single event, we could dynamically adjust our threshold constraints based on `ctx.tools.budget_remaining()`.
2. **Stateful Triage across Pillars:** If a producer's checkpoint fails or exhibits a soft anomaly, we would increase the sampling frequency of downstream lineage runs and data batches.
3. **Decoupled Overage Cap Optimization:** In this environment, the overage penalty caps at 20 points, whereas a missed fault incurs a larger relative TPR penalty. Under a strict, uncapped linear penalty, we would implement random skipping of AI infra tools (e.g. sample only 50% of feature writes) based on rolling probability of failure to maximize expected utility.
