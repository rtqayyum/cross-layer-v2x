# Cross-Layer V2X Cooperative Perception Scheduling

Reproducibility artifact for the proposed risk- and freshness-aware cross-layer scheduler for cooperative roadside perception under measured V2X service.

This repository contains only the final experiment code, frozen result summaries, tests, configuration, and provenance documentation. It contains no manuscript source, obsolete experiment versions, raw personal paths, or redistributed third-party datasets.

## What reviewers can verify immediately

The checked-in final results contain 4,212 paired controller runs over 39 real V2X-Seq-SPD sequences and 48 measured SEE-V2X traces. Reviewers can verify file integrity and regenerate every statistical table and Python result plot without downloading the raw datasets:

```bash
git clone https://github.com/rtqayyum/cross-layer-v2x.git
cd cross-layer-v2x
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./verify_artifact.sh
./analyze_frozen_results.sh
```

The analysis writes regenerated outputs to `results/reproduction/analysis/` and never overwrites `results/final/`.

## Main findings in the frozen artifact

| Outcome | Proposed controller | Difference from SafeSlice-P |
|---|---:|---:|
| Critical on-time ratio | 0.67397 | +0.02261 |
| Official car 3-D recall, IoU 0.3 | 0.10878 | +0.02862 |
| Official car 3-D AP, IoU 0.3 | 0.05278 | +0.00944 |
| P95 latency | 115.51 ms | -3.51 ms |
| Mean target PRBs | 56.74 | -15.60 |
| Background-SLA ratio | 1.000 | +0.32477 |
| Mean decision time | 0.155 ms | - |

The prespecified global result is deliberately retained as `DO_NOT_CLAIM_CONSTRAINED_SUPERIORITY`: risk-weighted utility is 0.00290 lower than SafeSlice-P and its lower non-inferiority confidence limit misses the -0.005 margin by 0.00021. The defensible conclusion is a strong safety-resource balance under a shared-service contract, not superiority on every metric.

## Repository structure

```text
.
├── config/                     Final campaign and dataset-access template
├── data/                       Empty ignored locations for official datasets
├── docs/                       Dataset mapping and reproducibility protocol
├── external/                   Empty ignored location for pinned official code
├── results/final/              Frozen reviewed outputs and run summaries
├── scripts/                    Data preparation, controller, evaluation, analysis
├── src/                        PCD reader used by the causal LiDAR pipeline
├── tests/                      Official-metric adapter tests
├── analyze_frozen_results.sh   Fast reviewer analysis
├── run_all.sh                  Full data-to-results reproduction
└── verify_artifact.sh          Checksums and public-path/secret scan
```

The machine identifier `proposed` denotes the paper's proposed controller. Where2comm-P, CodeFilling-P, and SafeSlice-P are mechanism-aligned paper projections under a common object-message interface; they are not claimed as official trained-model reproductions.

## Datasets and exact use

Raw data are not redistributed because the releases are large and separately governed. Obtain them only from the official sources and retain evidence of authorized research access. See [`docs/DATASETS.md`](docs/DATASETS.md) for the exact files, fields, transformations, and evaluation boundary.

| Source | Part used | Role in this study |
|---|---|---|
| [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq) | Sequential Perception Dataset metadata, calibration, cooperative labels, and infrastructure LiDAR | 6,532 synchronized real-road pairs; causal hypotheses; offline 3-D ground truth |
| [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) | Official evaluator core | Oriented 3-D IoU, recall, and AP logic |
| [SEE-V2X](https://github.com/UCR-CISL/SEE-V2X) | Emulating Applications records and SCI | Measured latency, loss, goodput, MCS, PRB, retransmission, and occupancy; 48 traces and 16 held-out groups |

V2X-Seq supplies the application/perception observations. SEE-V2X independently supplies measured network service observations. They were not captured together. The deterministic assignment and replay combine them without claiming joint capture or physical-radio actuation.

## Full reproduction from official data

Download these three official archives and keep the infrastructure `.z01` beside its `.zip` file:

- V2X-Seq SPD metadata archive: `V2X-Seq-SPD.zip`
- V2X-Seq infrastructure LiDAR archive: `V2X-Seq-SPD-infrastructure-side-velodyne.zip` and `.z01`
- SEE-V2X Emulating Applications archive: `emulating_applications.tar.gz`

Then run one command:

```bash
V2X_SEQ_ARCHIVE=/absolute/path/V2X-Seq-SPD.zip \
V2X_INFRA_ARCHIVE=/absolute/path/V2X-Seq-SPD-infrastructure-side-velodyne.zip \
SEE_V2X_ARCHIVE=/absolute/path/emulating_applications.tar.gz \
DATASET_ACCESS_CONFIRMED=yes \
./run_all.sh
```

The command creates a local virtual environment when needed, checks out the three pinned official repositories, indexes and freezes the outcome-blind real-road subset, runs the causal LiDAR detector, constructs the prospective lower service envelope, evaluates official AP/recall, executes the paired campaign, and regenerates all analyses. It is resumable but computationally intensive. Dataset files, external repositories, and reproduced outputs are ignored by Git.

## Integrity, testing, and frozen evidence

```bash
./verify_artifact.sh
make setup
make test
make analyze
```

Important frozen files:

- `results/final/run_summaries.parquet`: 4,212 paired run summaries.
- `results/final/analysis/clustered_comparisons.csv`: sequence-clustered effects, confidence intervals, and Holm-adjusted tests.
- `results/final/analysis/controller_means.csv`: controller-level KPIs.
- `results/final/analysis/timing_assurance.json`: decision-time evidence and claim boundary.
- `results/final/capacity_validation/scenario_validation.csv`: all 16 held-out capacity groups.
- `results/final/official_dair_metrics.json`: pinned official static metric result.

## Reproducibility and claim boundaries

- Real application observations: V2X-Seq-SPD.
- Real network observations: SEE-V2X.
- Counterfactual policy outcomes: deterministic trace replay.
- Independent statistical unit: road sequence, not individual run.
- No synthetic application rows.
- No physical-radio, WCET, official learned-baseline-equivalence, or universal-superiority claim.

The detailed protocol is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Required dataset citations

```bibtex
@inproceedings{v2x-seq,
  title={V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting},
  author={Yu, Haibao and Yang, Wenxian and Ruan, Hongzhi and Yang, Zhenwei and Tang, Yingjuan and Gao, Xu and Hao, Xin and Shi, Yifeng and Pan, Yifeng and Sun, Ning and Song, Juan and Yuan, Jirui and Luo, Ping and Nie, Zaiqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5486--5495},
  year={2023}
}

@inproceedings{dair-v2x,
  title={DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection},
  author={Yu, Haibao and Luo, Yizhen and Shu, Mao and Huo, Yiyi and Yang, Zebang and Shi, Yifeng and Guo, Zhenglong and Li, Hanyu and Hu, Xing and Yuan, Jirui and Nie, Zaiqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21361--21370},
  year={2022},
  doi={10.1109/CVPR52688.2022.02067}
}

@inproceedings{see-v2x,
  title={SEE-V2X: C-V2X Direct Communication Dataset: An Application-Centric Approach},
  author={Mo, Ruoshen and Wu, Bo and Tan, Zhaowei and Qiu, Hang},
  booktitle={Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems},
  series={SenSys '25},
  year={2025}
}
```

When using this repository, also cite the associated manuscript. During anonymous review, follow the target journal's artifact-anonymity policy before including a personal-account GitHub URL.

## License and redistribution

See [`LICENSE-NOTICE.md`](LICENSE-NOTICE.md). This repository does not grant redistribution rights for either dataset. No raw V2X-Seq or SEE-V2X content is committed.
