from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def simulate(seed=17, n=14000):
    """Synthetic coordinated device rings plus legitimate shared devices."""
    rng = np.random.default_rng(seed)
    ring_accounts = set(rng.choice(1000, 65, replace=False).tolist())
    device_pool = {a: (900 + a % 4 if a in ring_accounts else a) for a in range(1000)}
    events = []
    for t in range(n):
        account = int(rng.integers(1000))
        ring = account in ring_accounts
        # Legitimate kiosk sharing prevents equating any shared device with fraud.
        device = 850 if not ring and rng.random() < 0.035 else device_pool[account]
        amount = float(rng.lognormal(3.6, 0.65))
        risk = float(rng.beta(2, 7))
        probability = (
            (0.48 if ring else 0.006) + 0.03 * (amount > 100) + 0.02 * (risk > 0.5)
        )
        label = int(rng.random() < probability)
        events.append(
            dict(
                time=t,
                account=account,
                device=device,
                amount=amount,
                risk=risk,
                label=label,
                label_available=t + 180,
            )
        )
    return events


def point_in_time_features(events):
    """Update graph AFTER scoring each event. Consume only already-matured labels."""
    if any(events[i]["time"] >= events[i + 1]["time"] for i in range(len(events) - 1)):
        raise ValueError("Events must be strictly chronological")
    if any(e["label_available"] < e["time"] for e in events):
        raise ValueError("Labels cannot predate their events")
    seen = defaultdict(set)
    counts = defaultdict(int)
    mature = defaultdict(int)
    features = []
    evidence = []
    cursor = 0
    # Fixed delay in simulator, but sorting supports variable label availability too.
    matured_order = sorted(events, key=lambda e: e["label_available"])
    for event in events:
        t = event["time"]
        a = event["account"]
        d = event["device"]
        while (
            cursor < len(matured_order) and matured_order[cursor]["label_available"] < t
        ):
            past = matured_order[cursor]
            if past["time"] < t:
                mature[past["device"]] += past["label"]
            cursor += 1
        prior_accounts = seen[d] - {a}
        features.append(
            [
                np.log1p(event["amount"]),
                event["risk"],
                np.log1p(counts[a]),
                len(prior_accounts),
                mature[d],
            ]
        )
        evidence.append(
            dict(
                prior_shared_accounts=len(prior_accounts),
                matured_fraud_on_device=mature[d],
                prior_account_events=counts[a],
            )
        )
        seen[d].add(a)
        counts[a] += 1
    return np.array(features, float), evidence


def capacity_metrics(y, scores, k):
    y = np.asarray(y)
    scores = np.asarray(scores)
    if not 0 < k <= len(y):
        raise ValueError("Invalid review capacity")
    selected = np.argsort(-scores, kind="stable")[:k]
    caught = int(y[selected].sum())
    return dict(
        average_precision=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
        review_capacity=k,
        caught=caught,
        false_positives=k - caught,
        precision_at_capacity=caught / k,
        recall_at_capacity=caught / max(int(y.sum()), 1),
    )


def benchmark(events):
    x, evidence = point_in_time_features(events)
    times = np.array([e["time"] for e in events])
    available = np.array([e["label_available"] for e in events])
    y = np.array([e["label"] for e in events])
    # Train at 8000 using labels matured before then. Tune capacity policy on validation only.
    train = (times < 8000) & (available < 8000)
    validation = (times >= 8000) & (times < 10000) & (available < 10000)
    test = times >= 10000
    base = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=12, l2_regularization=10, random_state=17
    ).fit(x[train, :3], y[train])
    graph = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=12, l2_regularization=10, random_state=17
    ).fit(x[train], y[train])
    score_sets = dict(
        rules=(x[:, 0] > np.log1p(100)).astype(float) + x[:, 1],
        tabular=base.predict_proba(x[:, :3])[:, 1],
        graph=graph.predict_proba(x)[:, 1],
    )
    report = {
        "data": "SYNTHETIC; device-ring mechanism intentionally favours useful graph information",
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "test_rows": int(test.sum()),
        "test_prevalence": float(y[test].mean()),
        "label_delay": 180,
        "train_max_label_available": int(available[train].max()),
        "validation_max_label_available": int(available[validation].max()),
        "test_first_time": int(times[test].min()),
        "policy": "Review top 5% of transactions in each retrospective window; a batch budget, not a real-time threshold",
        "models": {},
    }
    for name, scores in score_sets.items():
        report["models"][name] = dict(
            validation=capacity_metrics(
                y[validation], scores[validation], int(validation.sum() * 0.05)
            ),
            test=capacity_metrics(y[test], scores[test], int(test.sum() * 0.05)),
        )
    # Choice fixed using validation average precision; do not select on final test.
    selected = max(
        report["models"],
        key=lambda name: report["models"][name]["validation"]["average_precision"],
    )
    report["validation_selected"] = selected
    indices = np.flatnonzero(test)
    ranked = indices[np.argsort(-score_sets[selected][test], kind="stable")[:20]]
    cases = [
        dict(
            event_index=int(i),
            account=events[i]["account"],
            device=events[i]["device"],
            score=float(score_sets[selected][i]),
            evidence=evidence[i],
            observed_label=int(y[i]),
        )
        for i in ranked
    ]
    return report, cases
