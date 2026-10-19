# Reusable API

```python
from fraudgraph.core import point_in_time_features
# Chronological dictionaries: time, account, device, amount, risk, label,
# label_available. Label availability cannot precede the event.
features, evidence = point_in_time_features(events)
```

Validate units, timestamps, missingness and evaluation assumptions. The simulator is not a substitute for representative labelled data.
