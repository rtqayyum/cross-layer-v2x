# Proposed controller confirmatory report

Runs: 4212; sequences: 39; traces: 48.
Macro sequence-average full-communication aligned oracle recall: 0.5579.
Decision: **DO_NOT_CLAIM_CONSTRAINED_SUPERIORITY**.

Primary comparator: `safeslice_paper_spec_projection`.

The prespecified claim is safety-constrained Pareto improvement: background-SLA feasibility, improved critical on-time delivery, and non-inferior official car AP and risk-weighted utility. It does not require or imply superiority on the older class-agnostic utility.

| Metric | Difference vs safeslice_paper_spec_projection | 95% CI | Holm p |
|---|---:|---:|---:|
| critical_on_time_ratio | 0.022608 | [0.013286, 0.031634] | 0.0001094 |
| contract_failure_rate | -0.026575 | [-0.03855, -0.015945] | 1.772e-05 |
| perception_recall | 0.17782 | [0.16356, 0.19255] | 2.838e-10 |
| perception_utility | 0.077729 | [0.065295, 0.090271] | 3.42e-10 |
| official_car_3d_recall_03 | 0.028618 | [0.021829, 0.036044] | 2.838e-10 |
| official_car_3d_ap_03 | 0.0094405 | [0.0058065, 0.013784] | 5.29e-06 |
| official_risk_weighted_utility | -0.0029037 | [-0.0052086, -0.00075652] | 0.02897 |
| p95_latency_ms | -3.5148 | [-3.7312, -3.2863] | 2.838e-10 |
| mean_target_prbs | -15.599 | [-16.707, -14.538] | 2.838e-10 |
| mean_fairness | 0.14764 | [0.13809, 0.1578] | 2.838e-10 |
| background_service_share | 0.1333 | [0.12442, 0.1425] | 2.838e-10 |
| background_sla_ratio | 0.32477 | [0.31432, 0.33553] | 2.838e-10 |
| fairness_pass_fraction | 0.85801 | [0.83111, 0.88458] | 2.838e-10 |

## Computational overhead

Measured around each controller decision on the execution host. The separate timing-assurance report adds CPU time, observed deadline margins, and a zero-exceedance probability bound. These remain empirical measurements, not a WCET certificate.

## Comparator qualification

`where2comm_object_proxy` and `codefilling_object_proxy` reproduce their published communication-selection ideas at object-message level under the same scheduler interface. They are not the authors’ learned feature-fusion networks and must not be reported as official reproductions. The SafeSlice comparator is likewise a paper-specification projection because no usable upstream implementation was available.
