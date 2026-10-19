import numpy as np
import pytest
from fraudgraph.core import *


def event(t, a, d, label=0, available=100):
    return dict(
        time=t,
        account=a,
        device=d,
        amount=10.0,
        risk=0.2,
        label=label,
        label_available=available,
    )


def test_graph_is_past_only_and_current_account_excluded():
    x, _ = point_in_time_features([event(0, 1, 7), event(1, 2, 7), event(2, 1, 7)])
    assert x[:, 3].tolist() == [0, 1, 1]


def test_delayed_labels_not_leaked():
    events = [
        event(0, 1, 7, label=1, available=2),
        event(1, 2, 7),
        event(2, 3, 7),
        event(3, 4, 7),
    ]
    x, _ = point_in_time_features(events)
    assert x[:, 4].tolist() == [0, 0, 0, 1]


def test_future_events_cannot_change_earlier_features():
    events = simulate(n=100)
    x, _ = point_in_time_features(events)
    assert np.array_equal(x[:50], point_in_time_features(events[:50])[0])


def test_capacity_and_imbalance():
    r = capacity_metrics([1, 0, 0, 1], [0.9, 0.8, 0.1, 0.7], 2)
    assert (
        r["caught"] == 1
        and r["false_positives"] == 1
        and r["recall_at_capacity"] == 0.5
    )


def test_unsorted_rejected():
    with pytest.raises(ValueError):
        point_in_time_features([event(2, 1, 1), event(1, 2, 1)])


def test_validation_labels_mature_before_test_selection():
    report, _ = benchmark(simulate())
    assert report["train_max_label_available"] < 8000
    assert report["validation_max_label_available"] < report["test_first_time"]
