#!/usr/bin/env python3
"""Build a leave-one-scenario-out SCI-calibrated service envelope.

Uses measured 100-ms application goodput as the response and measured SCI
MCS/PRB/retransmission features as predictors. A quantile gradient-boosted
tree is fit only on other scenarios for every held-out scenario.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# 3GPP TS 38.214 MCS table 1 spectral efficiencies (indices 0..28).
MCS_SE=np.array([.2344,.3066,.3770,.4902,.6016,.7402,.8770,1.0273,1.1758,1.3262,1.3281,1.4766,1.6953,1.9141,2.1602,2.4063,2.5703,2.5664,2.7305,3.0293,3.3223,3.6094,3.9023,4.2129,4.5234,4.8164,5.1152,5.3320,5.5547])
FEATURES=['radio_index','mean_mcs','mean_prbs','retransmission_rate','sci_messages','cbr_pct']
def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--application',type=Path,required=True);p.add_argument('--sci',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--validation-dir',type=Path,required=True);a=p.parse_args()
 app=pd.read_parquet(a.application);sci=pd.read_parquet(a.sci);app['scenario_id']=app.source_file.str.extract(r'emulating_applications/([^/]+/[^/]+/intersection_[^/]+)',expand=False);app['window_id']=(app.timestamp_us//100000).astype('Int64')
 # One rolling-goodput value per link/window, then aggregate the three links.
 link=(app.assign(goodput=lambda x:x.throughput_bps.where(x.throughput_bps>1,0)).groupby(['scenario_id','trace_id','window_id'],observed=True).agg(link_goodput=('goodput','max'),cbr_pct=('channel_busy_pct','median')).reset_index())
 aw=link.groupby(['scenario_id','window_id'],observed=True).agg(measured_goodput_bps=('link_goodput','sum'),cbr_pct=('cbr_pct','median')).reset_index()
 sci['window_id']=(sci.rx_timestamp_us//100000).astype('int64');sci['mcs_idx']=sci.mcs_idx.clip(0,len(MCS_SE)-1).astype(int);sci['spectral_efficiency']=MCS_SE[sci.mcs_idx]
 keys=['scenario_id','window_id','rx_timestamp_us']+[x for x in ['sf_idx','pssch_start_idx','N_x_id'] if x in sci]
 unique=sci.drop_duplicates(keys).copy();unique['resource_score']=unique.spectral_efficiency*unique.nof_prb_pssch*(1-unique.retransmission.clip(0,1))
 sw=unique.groupby(['scenario_id','window_id'],observed=True).agg(radio_index=('resource_score','sum'),mean_mcs=('mcs_idx','mean'),mean_prbs=('nof_prb_pssch','mean'),retransmission_rate=('retransmission','mean'),sci_messages=('mcs_idx','size')).reset_index()
 windows=aw.merge(sw,on=['scenario_id','window_id'],how='inner').replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['measured_goodput_bps'])
 if windows.scenario_id.nunique()<8:raise SystemExit('SCI capacity gate failed: fewer than 8 aligned scenarios')
 predictions=[]
 for scenario,test in windows.groupby('scenario_id',sort=True):
  train=windows[windows.scenario_id!=scenario];proper=train[train.measured_goodput_bps>0].copy()
  if len(proper)>60000:proper=proper.sample(60000,random_state=20261006)
  model=GradientBoostingRegressor(loss='quantile',alpha=.10,n_estimators=120,max_depth=3,min_samples_leaf=30,learning_rate=.05,random_state=20261006)
  model.fit(proper[FEATURES],np.log1p(proper.measured_goodput_bps));ordered=test.sort_values('window_id');cut=max(100,int(.20*len(ordered)));calibration=ordered.iloc[:cut];evaluation=ordered.iloc[cut:]
  base_pred=np.expm1(model.predict(ordered[FEATURES]));scores=base_pred-ordered.measured_goodput_bps.to_numpy()
  # Adaptive conformal correction using prior observations only.  A 95%
  # higher-interpolated rolling residual quantile responds to within-scenario
  # drift without seeing the current or any future measured-goodput value.
  target_coverage=.95;rolling_window=5000
  qhat=pd.Series(scores).shift(1).rolling(rolling_window,min_periods=min(cut,rolling_window)).quantile(target_coverage,interpolation='higher').to_numpy()
  pred=(base_pred[cut:]-qhat[cut:]).clip(0)
  part=evaluation.copy();part['capacity_bps']=pred;part['calibration_windows']=cut;part['conformal_history_windows']=np.minimum(np.arange(cut,len(ordered)),rolling_window);predictions.append(part)
 pred=pd.concat(predictions,ignore_index=True);rho,pv=spearmanr(pred.capacity_bps,pred.measured_goodput_bps);coverage=float((pred.measured_goodput_bps>=pred.capacity_bps).mean());mae=float(np.mean(np.abs(pred.capacity_bps-pred.measured_goodput_bps)))
 mapping=pred[['scenario_id','window_id','capacity_bps']];result=app.merge(mapping,on=['scenario_id','window_id'],how='inner',validate='many_to_one');result.capacity_bps=result.capacity_bps.clip(lower=0);result['capacity_model']='LOSO_SCI_MCS_PRB_Q10_CQR_V2'
 a.output.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(a.output,index=False);a.validation_dir.mkdir(parents=True,exist_ok=True);pred.to_parquet(a.validation_dir/'aligned_window_predictions.parquet',index=False)
 scenario=pred.groupby('scenario_id').apply(lambda x:pd.Series({'windows':len(x),'spearman_rho':spearmanr(x.capacity_bps,x.measured_goodput_bps).statistic,'lower_bound_coverage':(x.measured_goodput_bps>=x.capacity_bps).mean(),'median_capacity_bps':x.capacity_bps.median()}),include_groups=False).reset_index();scenario.to_csv(a.validation_dir/'scenario_validation.csv',index=False)
 sample=pred.sample(min(50000,len(pred)),random_state=20261006);fig,ax=plt.subplots(figsize=(6.5,5));ax.hexbin(sample.measured_goodput_bps/1e6,sample.capacity_bps/1e6,gridsize=55,mincnt=1,cmap='viridis');limit=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,limit],[0,limit],'r--',lw=1);ax.set(xlabel='Measured goodput (Mbit/s)',ylabel='Lower service bound (Mbit/s)',title=f'Held-out SCI lower-bound validation: rho={rho:.3f}, coverage={coverage:.3f}');fig.tight_layout();fig.savefig(a.validation_dir/'capacity_validation_scatter.png',dpi=250);fig.savefig(a.validation_dir/'capacity_validation_scatter.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));ax.bar(np.arange(len(scenario)),scenario.lower_bound_coverage);ax.axhline(.9,color='r',ls='--',label='nominal 90%');ax.set_xticks(np.arange(len(scenario)),scenario.scenario_id.str.replace('/','\n',regex=False),rotation=70,ha='right',fontsize=6);ax.set(ylabel='P(measured goodput >= bound)',ylim=(0,1.02),title='Lower service-bound coverage by held-out scenario');ax.legend();fig.tight_layout();fig.savefig(a.validation_dir/'scenario_coverage.png',dpi=250);fig.savefig(a.validation_dir/'scenario_coverage.pdf');plt.close(fig)
 scenario_pass=int((scenario.lower_bound_coverage>=.85).sum());status='PASS_SCI_CAPACITY_VALIDATION' if np.isfinite(rho) and rho>=.30 and pv<.01 and coverage>=.90 and scenario_pass>=14 and pred.capacity_bps.nunique()>=100 else 'FAIL_SCI_CAPACITY_VALIDATION'
 receipt={'status':status,'bound_direction':'LOWER_SERVICE_BOUND','method':'leave-one-scenario-out Q10 GradientBoostingRegressor plus causal rolling 95% conformal correction; first 20% initializes calibration and each final-80% prediction uses only prior residuals','mcs_reference':'3GPP TS 38.214 MCS table 1 spectral efficiency','aligned_evaluation_windows':len(pred),'calibration_fraction':.20,'conformal_target_coverage':target_coverage,'conformal_history_max_windows':rolling_window,'causal_no_future_leakage':True,'scenarios':int(pred.scenario_id.nunique()),'scenario_coverage_pass_count':scenario_pass,'scenario_coverage_required':14,'spearman_rho':float(rho),'spearman_p':float(pv),'lower_bound_coverage':coverage,'coverage_acceptance_floor':.90,'mae_bps':mae,'capacity_unique_values':int(pred.capacity_bps.nunique()),'capacity_quantiles_bps':{str(q):float(pred.capacity_bps.quantile(q)) for q in [0.01,.1,.5,.9,.99]},'application_sha256':sha(a.application),'sci_sha256':sha(a.sci),'output_sha256':sha(a.output),'physical_radio_claim':False,'limitations':'Prospectively calibrated trace-level lower service bound, not a 3GPP link-level simulator, WCET proof, or OTA validation.'}
 (a.validation_dir/'sci_capacity_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return 0 if status.startswith('PASS') else 3
if __name__=='__main__':raise SystemExit(main())
