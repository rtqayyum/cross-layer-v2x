# Dataset provenance and use

Raw datasets are intentionally not included. They are large, separately distributed, and subject to their providers' access terms. The scripts fail closed unless the user confirms authorized research access.

## V2X-Seq Sequential Perception Dataset

- Official project: <https://github.com/AIR-THU/DAIR-V2X-Seq>
- Official dataset entry point: <https://github.com/AIR-THU/DAIR-V2X>
- Paper: H. Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting," CVPR 2023.

### Parts used

1. `V2X-Seq-SPD.zip`: cooperative pair metadata, timestamps, sequence IDs, calibration, and cooperative 3-D labels.
2. `V2X-Seq-SPD-infrastructure-side-velodyne.zip` with its adjacent split volume: infrastructure LiDAR measurements.
3. The official DAIR-V2X evaluator code, checked out at commit `c885c54af0c34bc515fa9ca8b5e8fda76a15462c`.

### How it is used

The metadata archive is indexed without modifying the source. An outcome-blind deterministic selection freezes 6,532 synchronized pairs from 39 source sequences. Infrastructure LiDAR is processed frame by frame by the causal geometric detector. The resulting hypotheses are aligned into the vehicle frame using source calibration. Cooperative labels remain unavailable to the scheduler and are read only during offline official 3-D IoU, recall, and AP evaluation.

## SEE-V2X

- Official repository and access instructions: <https://github.com/UCR-CISL/SEE-V2X>
- Paper: R. Mo, B. Wu, Z. Tan, and H. Qiu, "SEE-V2X: C-V2X Direct Communication Dataset: An Application-Centric Approach," ACM SenSys 2025.

### Parts used

The study uses the `Emulating Applications` release, including perception-only and BSM-plus-perception measurements from indoor extended-duration and outdoor-intersection captures. The parser reads application transmit/receive CSV files and `sci.csv` fields containing MCS, PRB, retransmission, message-count, and channel-occupancy information.

### How it is used

The release is normalized into 48 receiver-transmitter traces and 16 held-out scenario groups. Application records provide measured loss, latency, and goodput. SCI records provide cross-layer radio features. Leave-one-scenario-out quantile regression and past-only conformal residuals produce a lower service envelope over 249,252 held-out 100-ms windows. The trace is used for deterministic counterfactual replay; it is not presented as an over-the-air execution of the proposed scheduler.

## Citation requirement

Any paper or derivative study using this artifact should cite both dataset papers, not only this repository. Ready-to-copy BibTeX entries are provided in the main `README.md`.

