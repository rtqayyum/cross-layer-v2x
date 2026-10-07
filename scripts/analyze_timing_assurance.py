#!/usr/bin/env python3
"""Empirical controller timing assurance with an explicit non-WCET boundary."""
from __future__ import annotations
import argparse,json,platform
from pathlib import Path
import numpy as np,pandas as pd

def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--deadline-ms',type=float,default=100.);a=p.parse_args();d=pd.read_parquet(a.results);a.output_dir.mkdir(parents=True,exist_ok=True)
 required={'decision_time_ms_max','decision_cpu_time_ms_max','decision_count'}
 if not required.issubset(d):raise SystemExit(f"timing columns missing: {sorted(required-set(d.columns))}")
 rows=[]
 for ctl,g in d.groupby('controller_id',sort=True):
  n=int(g.decision_count.sum());wall_max=float(g.decision_time_ms_max.max());cpu_max=float(g.decision_cpu_time_ms_max.max());exceed_runs=int((g.decision_time_ms_max>a.deadline_ms).sum())
  rows.append({'controller_id':ctl,'decisions':n,'deadline_ms':a.deadline_ms,'wall_time_run_mean_ms':float(g.decision_time_ms_mean.mean()),'wall_time_observed_max_ms':wall_max,'cpu_time_run_mean_ms':float(g.decision_cpu_time_ms_mean.mean()),'cpu_time_observed_max_ms':cpu_max,'minimum_observed_margin_ms':a.deadline_ms-wall_max,'runs_with_any_deadline_exceedance':exceed_runs,'zero_exceedance_probability_upper_95':float(1-.05**(1/max(n,1))) if exceed_runs==0 else np.nan,'complexity':'O(K log K), K <= 3N object-mode candidates','empirical_only':True})
 out=pd.DataFrame(rows);out.to_csv(a.output_dir/'timing_assurance.csv',index=False)
 passed=bool((out.runs_with_any_deadline_exceedance==0).all());receipt={'status':'PASS_EMPIRICAL_TIMING_ASSURANCE' if passed else 'FAIL_EMPIRICAL_TIMING_ASSURANCE','deadline_ms':a.deadline_ms,'controllers':len(out),'total_decisions':int(out.decisions.sum()),'minimum_margin_ms':float(out.minimum_observed_margin_ms.min()),'host':{'platform':platform.platform(),'python':platform.python_version(),'processor':platform.processor() or 'not reported'},'claim_boundary':'Repeated wall/CPU timing and a zero-exceedance binomial upper bound; this is not static WCET certification or a real-time OS guarantee.'}
 (a.output_dir/'timing_assurance.json').write_text(json.dumps(receipt,indent=2)+'\n')
 lines=['# Timing assurance','',f"Status: **{receipt['status']}**. Scheduling interval: {a.deadline_ms:g} ms.",'',f"Across {receipt['total_decisions']:,} decisions, the minimum observed wall-clock margin was {receipt['minimum_margin_ms']:.3f} ms.",'','`zero_exceedance_probability_upper_95` is the exact one-sided 95% binomial upper bound when no exceedance is observed. It quantifies empirical evidence only; it is not a worst-case execution-time proof. The selector has O(K log K) complexity for at most three payload modes per object.','','| Controller | Decisions | Observed max (ms) | Minimum margin (ms) | 95% upper exceedance probability |','|---|---:|---:|---:|---:|']
 for r in out.itertuples(index=False):lines.append(f"| {r.controller_id} | {r.decisions} | {r.wall_time_observed_max_ms:.6g} | {r.minimum_observed_margin_ms:.6g} | {r.zero_exceedance_probability_upper_95:.6g} |")
 (a.output_dir/'timing_assurance.md').write_text('\n'.join(lines)+'\n');print(json.dumps(receipt,indent=2));return 0 if passed else 4
if __name__=='__main__':raise SystemExit(main())
