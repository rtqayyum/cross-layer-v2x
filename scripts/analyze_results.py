#!/usr/bin/env python3
"""Confirmatory sequence-clustered analysis for the proposed controller."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scipy.stats import wilcoxon

METRICS={"critical_on_time_ratio":True,"contract_failure_rate":False,"perception_recall":True,"perception_utility":True,"official_car_3d_recall_03":True,"official_car_3d_ap_03":True,"official_risk_weighted_utility":True,"p95_latency_ms":False,"mean_target_prbs":False,"mean_fairness":True,"background_service_share":True,"background_sla_ratio":True,"fairness_pass_fraction":True}

def save(fig,o,n):fig.savefig(o/f"{n}.png",dpi=250,bbox_inches='tight');fig.savefig(o/f"{n}.pdf",bbox_inches='tight');plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();d=pd.read_parquet(a.results);o=a.output_dir;o.mkdir(parents=True,exist_ok=True)
 if set(d.network_provenance)!={"MEASURED_SEE_V2X_CROSSFIT_CAPACITY_REPLAY"}:raise SystemExit('provenance gate failed')
 focus=[x for x in ['proportional_fair_fifo','earliest_deadline_first','lyapunov_dpp','where2comm_object_proxy','codefilling_object_proxy','safeslice_style_projection','safeslice_adaptive_payload','safeslice_paper_spec_projection','proposed_no_radio','proposed_no_fairness','proposed'] if x in set(d.controller_id)];means=d.groupby('controller_id')[list(METRICS)].mean().loc[focus];means.to_csv(o/'controller_means.csv')
 keys=['sequence_id','background_streams','repetition','trace_id'];cp=d[d.controller_id=='proposed'];rows=[];rng=np.random.default_rng(20261006)
 for base in [x for x in ['lyapunov_dpp','where2comm_object_proxy','codefilling_object_proxy','safeslice_paper_spec_projection','safeslice_adaptive_payload','safeslice_style_projection','proposed_no_radio','proposed_no_fairness'] if x in set(d.controller_id)]:
  paired=cp.merge(d[d.controller_id==base],on=keys,suffixes=('_cp','_base'),validate='one_to_one')
  for metric,higher in METRICS.items():
   seq=(paired.assign(effect=paired[f'{metric}_cp']-paired[f'{metric}_base']).groupby('sequence_id')["effect"].mean()).dropna();boot=rng.choice(seq.to_numpy(),(5000,len(seq)),replace=True).mean(1);p=1. if np.allclose(seq,0) else wilcoxon(seq,zero_method='pratt').pvalue
   rows.append({'baseline_id':base,'metric':metric,'mean_difference':seq.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'p_value':p,'favorable_sequence_fraction':np.mean(seq>0) if higher else np.mean(seq<0),'higher_is_better':higher})
 comp=pd.DataFrame(rows);comp['holm_p']=np.nan
 ordered=comp.p_value.sort_values();running=0.
 for rank,(idx,pvalue) in enumerate(ordered.items()):running=max(running,min(1.,pvalue*(len(ordered)-rank)));comp.loc[idx,'holm_p']=running
 comp.to_csv(o/'clustered_comparisons.csv',index=False)
 labels=[x.replace('_',' ').title() for x in focus];colors=list(plt.cm.tab10(np.linspace(0,1,len(focus))))
 fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
 for ax,m in zip(axs.flat,['critical_on_time_ratio','official_car_3d_ap_03','official_risk_weighted_utility','background_sla_ratio']):ax.bar(range(len(focus)),means[m],color=colors);ax.set_xticks(range(len(focus)),labels,rotation=25,ha='right');ax.set_title(m.replace('_',' ').title());ax.grid(axis='y',alpha=.25)
 save(fig,o,'01_primary_outcomes')
 fig,axs=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
 grouped=d.groupby(['controller_id','background_streams']).mean(numeric_only=True)
 for ax,m in zip(axs.flat,['critical_on_time_ratio','perception_utility','p95_latency_ms','mean_target_prbs','mean_fairness','fairness_pass_fraction']):
  for ctl,col in zip(focus,colors):ax.plot(sorted(d.background_streams.unique()),[grouped.loc[(ctl,x),m] for x in sorted(d.background_streams.unique())],marker='o',color=col,label=ctl.replace('_',' '));ax.set_title(m.replace('_',' ').title());ax.grid(alpha=.25)
 axs.flat[0].legend(fontsize=7);save(fig,o,'02_load_robustness')
 fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 for ax,(x,y) in zip(axs,[('mean_target_prbs','critical_on_time_ratio'),('mean_fairness','critical_on_time_ratio'),('background_sla_ratio','perception_utility')]):
  for ctl,col in zip(focus,colors):ax.scatter(means.loc[ctl,x],means.loc[ctl,y],color=col,s=80);ax.annotate(ctl.replace('_',' '),(means.loc[ctl,x],means.loc[ctl,y]),fontsize=7);ax.set_xlabel(x.replace('_',' '));ax.set_ylabel(y.replace('_',' '));ax.grid(alpha=.25)
 save(fig,o,'03_pareto_frontiers')
 fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 for ax,m in zip(axs,['official_car_3d_recall_03','official_car_3d_ap_03','official_risk_weighted_utility']):
  for ctl,col in zip(focus,colors):ax.plot(sorted(d.background_streams.unique()),[grouped.loc[(ctl,x),m] for x in sorted(d.background_streams.unique())],marker='o',color=col,label=ctl.replace('_',' '))
  ax.set(xlabel='Background streams',title=m.replace('_',' ').title());ax.grid(alpha=.25)
 axs[0].legend(fontsize=6);save(fig,o,'04_official_metrics_by_load')
 heat_metrics=['critical_on_time_ratio','official_car_3d_ap_03','official_risk_weighted_utility','background_sla_ratio','mean_fairness'];heat=means[heat_metrics].copy()
 for m in heat_metrics:
  lo,hi=heat[m].min(),heat[m].max();heat[m]=0. if hi==lo else (heat[m]-lo)/(hi-lo)
 fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(heat.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(heat_metrics)),[m.replace('_',' ') for m in heat_metrics],rotation=30,ha='right');ax.set_yticks(range(len(focus)),labels);fig.colorbar(im,ax=ax,label='Within-metric normalized value');ax.set_title('Controller KPI profile');save(fig,o,'05_controller_kpi_heatmap')
 primary_id='safeslice_paper_spec_projection' if 'safeslice_paper_spec_projection' in set(comp.baseline_id) else 'safeslice_adaptive_payload';primary=comp[comp.baseline_id==primary_id]
 def low(metric):return float(primary.query('metric==@metric').ci95_low.iloc[0])
 fairness_ok=bool(means.loc['proposed','fairness_pass_fraction']>=.95);critical_superiority=low('critical_on_time_ratio')>0;official_ap_noninferiority=low('official_car_3d_ap_03')>=-.005;risk_utility_noninferiority=low('official_risk_weighted_utility')>=-.005
 passes=fairness_ok and critical_superiority and official_ap_noninferiority and risk_utility_noninferiority
 decision='PASS_SAFETY_CONSTRAINED_PARETO' if passes else 'DO_NOT_CLAIM_CONSTRAINED_SUPERIORITY'
 report=['# Proposed controller confirmatory report','',f"Runs: {len(d)}; sequences: {d.sequence_id.nunique()}; traces: {d.trace_id.nunique()}.",f"Macro sequence-average full-communication aligned oracle recall: {d.oracle_full_recall.mean():.4f}.",f"Decision: **{decision}**.",'',f'Primary comparator: `{primary_id}`.','', 'The prespecified claim is safety-constrained Pareto improvement: background-SLA feasibility, improved critical on-time delivery, and non-inferior official car AP and risk-weighted utility. It does not require or imply superiority on the older class-agnostic utility.','',f'| Metric | Difference vs {primary_id} | 95% CI | Holm p |','|---|---:|---:|---:|']
 for r in primary.itertuples():report.append(f"| {r.metric} | {r.mean_difference:.5g} | [{r.ci95_low:.5g}, {r.ci95_high:.5g}] | {r.holm_p:.4g} |")
 show=primary[primary.metric.isin(['critical_on_time_ratio','official_car_3d_ap_03','official_risk_weighted_utility','background_sla_ratio'])].copy();fig,ax=plt.subplots(figsize=(8,4.5));y=np.arange(len(show));ax.errorbar(show.mean_difference,y,xerr=[show.mean_difference-show.ci95_low,show.ci95_high-show.mean_difference],fmt='o',capsize=4);ax.axvline(0,color='black',lw=1);ax.set_yticks(y,show.metric.str.replace('_',' '));ax.set(xlabel=f'Proposed minus {primary_id}',title='Sequence-clustered effects with 95% bootstrap intervals');ax.grid(axis='x',alpha=.25);save(fig,o,'06_primary_effect_forest')
 if 'decision_time_ms_p95' in d: 
  cols=['decision_time_ms_mean','decision_time_ms_p95','decision_time_ms_p99','decision_time_ms_max','decision_time_us_per_candidate']+[x for x in ['decision_cpu_time_ms_mean','decision_cpu_time_ms_p99','decision_cpu_time_ms_max'] if x in d];overhead=d.groupby('controller_id')[cols].agg(['mean','median','max']);overhead.to_csv(o/'controller_computational_overhead.csv');report += ['', '## Computational overhead', '', 'Measured around each controller decision on the execution host. The separate timing-assurance report adds CPU time, observed deadline margins, and a zero-exceedance probability bound. These remain empirical measurements, not a WCET certificate.']
 report += ['', '## Comparator qualification', '', '`where2comm_object_proxy` and `codefilling_object_proxy` reproduce their published communication-selection ideas at object-message level under the same scheduler interface. They are not the authors’ learned feature-fusion networks and must not be reported as official reproductions. The SafeSlice comparator is likewise a paper-specification projection because no usable upstream implementation was available.']
 (o/'confirmatory_report.md').write_text('\n'.join(report)+'\n');manifest={'status':'ANALYSIS_COMPLETE','decision':decision,'criteria':{'background_sla_feasible':fairness_ok,'critical_delivery_superiority':critical_superiority,'official_ap_noninferiority_margin_0.005':official_ap_noninferiority,'risk_utility_noninferiority_margin_0.005':risk_utility_noninferiority},'runs':len(d),'sequences':int(d.sequence_id.nunique()),'physical_radio_claim':False};(o/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
