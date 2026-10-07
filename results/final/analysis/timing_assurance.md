# Timing assurance

Status: **PASS_EMPIRICAL_TIMING_ASSURANCE**. Scheduling interval: 100 ms.

Across 705,456 decisions, the minimum observed wall-clock margin was 99.245 ms.

`zero_exceedance_probability_upper_95` is the exact one-sided 95% binomial upper bound when no exceedance is observed. It quantifies empirical evidence only; it is not a worst-case execution-time proof. The selector has O(K log K) complexity for at most three payload modes per object.

| Controller | Decisions | Observed max (ms) | Minimum margin (ms) | 95% upper exceedance probability |
|---|---:|---:|---:|---:|
| codefilling_object_proxy | 78384 | 0.574125 | 99.4259 | 3.82179e-05 |
| earliest_deadline_first | 78384 | 0.180583 | 99.8194 | 3.82179e-05 |
| lyapunov_dpp | 78384 | 0.293292 | 99.7067 | 3.82179e-05 |
| proportional_fair_fifo | 78384 | 0.105083 | 99.8949 | 3.82179e-05 |
| proposed | 78384 | 0.754625 | 99.2454 | 3.82179e-05 |
| proposed_no_fairness | 78384 | 0.279625 | 99.7204 | 3.82179e-05 |
| proposed_no_radio | 78384 | 0.28375 | 99.7163 | 3.82179e-05 |
| safeslice_paper_spec_projection | 78384 | 0.273666 | 99.7263 | 3.82179e-05 |
| where2comm_object_proxy | 78384 | 0.279208 | 99.7208 | 3.82179e-05 |
