# FraudGraph: methodology

Historical graph features count previously seen distinct accounts on a device and already-matured fraud labels. Identifiers are not numeric predictors. Train at tick 8,000 using labels matured strictly before that cutoff; validation covers 8,000–9,999; test starts at 10,000. Models are frozen after training. During validation/test, online features can incorporate earlier events and labels that have since matured, representing a live feedback system rather than a static holdout feature matrix. Select the model by validation AP. Evaluate the top 5% in each retrospective batch under a fixed review budget.

## Assumptions and limits

The data generator deliberately contains graph-related fraud structure and cannot establish general real-world superiority. This is graph feature engineering, not a graph neural network. Shared devices can be legitimate, and investigator evidence is not causal proof or a guilt determination. Batch top-k ranking uses a complete window of scores and is not an online threshold policy. Case labels are shown only as retrospective outcomes. No analyst decisions or transactions are sent anywhere.

## Extension plan

Evaluate public transaction data, graph ablations, concept drift, investigation delay, account-disjoint generalisation and precision intervals.
