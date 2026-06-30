# Judge Bias Report

- Total judged: 30
- Position bias rate: 0.0%
- Verbosity bias: 42.9%
- Cohen's kappa sample: 0.000

## Mitigation Strategy

Use swap-and-average for every pairwise comparison, log run1/run2 disagreement, and calibrate judge output against human labels before trusting it for production gates.
