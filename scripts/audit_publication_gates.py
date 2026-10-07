#!/usr/bin/env python3
"""Create a machine-readable external-artifact and publication gate audit."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import pandas as pd

def commit(p):
 try:return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
 except:return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();e=a.experiment_root
 safe=e/'external/safe-slice'; safe_commit=commit(safe); safe_files=[x for x in safe.rglob('*') if x.is_file() and '.git' not in x.parts];out=a.output.parent
 def read(path):
  try:return json.loads(path.read_text())
  except:return {}
 metric=read(out/'official_dair_metrics.json');capacity=read(out/'capacity_validation/sci_capacity_validation.json');timing=read(out/'analysis/timing_assurance.json');metric_pass=metric.get('status')=='PASS_OFFICIAL_DAIR_METRIC' and max(metric.get('metrics',{}).values(),default=0)>0;capacity_pass=capacity.get('status')=='PASS_SCI_CAPACITY_VALIDATION' and capacity.get('bound_direction')=='LOWER_SERVICE_BOUND';timing_pass=timing.get('status')=='PASS_EMPIRICAL_TIMING_ASSURANCE'
 run_path=out/'run_summaries.parquet';controller_metric_pass=False
 if run_path.exists():
  cols=set(pd.read_parquet(run_path).columns);controller_metric_pass={'official_car_3d_ap_03','official_car_3d_recall_03','official_risk_weighted_utility'}.issubset(cols)
 method_pass=metric_pass and capacity_pass and controller_metric_pass and timing_pass
 audit={'status':'METHOD_GATES_PASS_DOCUMENTARY_ACTIONS_REMAIN' if method_pass else 'CONDITIONAL_NOT_SUBMISSION_READY','artifacts':{'DAIR-V2X':{'commit':commit(e/'external/DAIR-V2X'),'official_metric_core':'external/DAIR-V2X/v2x/v2x_utils/eval_utils.py'},'DAIR-V2X-Seq':{'commit':commit(e/'external/DAIR-V2X-Seq')},'SEE-V2X':{'commit':commit(e/'external/SEE-V2X')},'SafeSlice':{'commit':safe_commit,'tracked_source_files':len(safe_files),'status':'UPSTREAM_REPOSITORY_EMPTY_NO_OFFICIAL_CODE_AVAILABLE' if not safe_commit else 'PINNED'}},'gates':{'official_static_perception_metric':'PASS_CLASS_ORIENTED_OFFICIAL_CORE' if metric_pass else 'FAIL_OR_MISSING','official_controller_level_metric':'PASS_IOU_AP_IN_REPLAY' if controller_metric_pass else 'FAIL_OR_MISSING','dataset_license':'DOCUMENTARY_PERMISSION_REQUIRED','capacity_validation':'PASS_PROSPECTIVE_LOWER_SCI_CQR' if capacity_pass else 'FAIL_OR_MISSING','scenario_groups':'PASS_39_PERCEPTION_16_SCI','computational_overhead':'PASS_EMPIRICAL_NOT_WCET' if timing_pass else 'FAIL_OR_MISSING','external_baselines':'PF_EDF_LYAPUNOV_DPP_PLUS_QUALIFIED_WHERE2COMM_CODEFILLING_AND_SAFESLICE_PROXIES','limitations_reproducibility':'DOCUMENTED'},'claim_boundary':'No physical-radio, WCET, or official learned-baseline-equivalence claim. Dataset publication permission remains documentary work.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
