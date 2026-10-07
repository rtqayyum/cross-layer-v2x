# Reproducibility protocol

## Independent unit and pairing

The road sequence is the independent statistical unit. Every controller receives the same sequence, network-trace assignment, background load, repetition, payload serialization, and deterministic random draw within a paired block. Repeated loads and traces are not treated as independent road scenes.

## Causality controls

- Perception decisions use only the current detector output and retained past state.
- Capacity calibration uses only residuals observed before the current 100-ms window.
- Ground-truth labels and IoU candidates are evaluation-only artifacts.
- Every source and derived table is hash bound.

## Frozen campaign

- 39 V2X-Seq-SPD sequences and 6,532 synchronized pairs.
- 540,434 causal detector hypotheses.
- 48 SEE-V2X traces grouped into 16 held-out scenarios.
- Four concurrent-load settings, three repetitions, and nine controllers.
- 4,212 paired runs and 705,456 measured controller decisions.
- 5,000 sequence-cluster bootstrap replications.

## Claim boundary

The frozen analysis decision is `DO_NOT_CLAIM_CONSTRAINED_SUPERIORITY` because the risk-utility non-inferiority condition misses its lower confidence margin. The supported claim is an improved safety-resource balance with full observed background-SLA satisfaction, not superiority on every utility. The timing result is empirical Python wall-clock and CPU timing, not WCET certification. The network evaluation is measured-trace replay, not a physical O-RAN experiment.

