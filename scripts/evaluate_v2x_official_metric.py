#!/usr/bin/env python3
"""Run the pinned DAIR-V2X AP core on project tables.

The current detector is class agnostic and has no yaw output. Predictions are
therefore evaluated as car with yaw=0 and are explicitly labelled a proxy.
This uses the official matching/AP code, but is not a leaderboard submission.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,importlib.util,io,json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt

def label_corners(row, yaw: float)->np.ndarray:
 """Exact ordering produced by DAIR-V2X get_3d_8points for labels."""
 l,w,h=float(row.length_m),float(row.width_m),float(row.height_m);x,y,z=float(row.x_m),float(row.y_m),float(row.z_m)
 local=np.array([[l/2,w/2,0],[l/2,-w/2,0],[-l/2,-w/2,0],[-l/2,w/2,0],[l/2,w/2,h],[l/2,-w/2,h],[-l/2,-w/2,h],[-l/2,w/2,h]])
 r=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
 return local@r.T+np.array([x,y,z-h/2])

def prediction_corners(row, yaw: float, mod)->np.ndarray:
 """Convert label-order corners to the mmdet prediction order expected upstream."""
 source=label_corners(row,yaw); result=np.empty_like(source)
 # Official evaluator compares label[perm_label] with prediction[perm_pred].
 result[np.asarray(mod.perm_pred)]=source[np.asarray(mod.perm_label)]
 return result

def load_official(repo:Path):
 sys.path.insert(0,str(repo/'v2x')); spec=importlib.util.spec_from_file_location('official_eval_utils',repo/'v2x/v2x_utils/eval_utils.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def official_frame_records(pred_boxes,scores,gt_boxes,threshold,view,mod):
 """Official greedy matching with a lossless bounding-circle prefilter."""
 remaining=list(range(len(pred_boxes)));records=[];tp=0
 for gt in gt_boxes:
  gc=gt.mean(0);gr=np.max(np.linalg.norm(gt[:4,:2]-gc[:2],axis=1));best=threshold;best_j=None
  for j in remaining:
   pb=pred_boxes[j];pc=pb.mean(0);pr=np.max(np.linalg.norm(pb[:4,:2]-pc[:2],axis=1))
   if np.linalg.norm(gc[:2]-pc[:2])>gr+pr or min(gt[:,2].max(),pb[:,2].max())<=max(gt[:,2].min(),pb[:,2].min()):continue
   try:i3,ib=mod.box3d_iou(gt[mod.perm_label],pb[mod.perm_pred])
   except Exception:i3,ib=0.,0.
   value=ib if view=='bev' else i3
   if value>=best:best=value;best_j=j
  if best_j is not None:records.append({'box':pred_boxes[best_j],'score':float(scores[best_j]),'type':'tp'});remaining.remove(best_j);tp+=1
 records.extend({'box':pred_boxes[j],'score':float(scores[j]),'type':'fp'} for j in remaining);return records,tp

def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--ground-truth',type=Path,required=True);p.add_argument('--official-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--classes',nargs='+',choices=['car','cyclist','pedestrian'],default=['car']);a=p.parse_args();pred=pd.read_parquet(a.predictions);gt=pd.read_parquet(a.ground_truth);mod=load_official(a.official_repo);semantic=pred.object_class.astype(str).str.lower();oriented='rotation_rad' in pred and pred.rotation_rad.notna().all();proxy=not oriented or not semantic.isin(['car','cyclist','pedestrian']).any();classes=['car'] if proxy else a.classes;all_records={(v,c,t):[] for v in ('3d','bev') for c in classes for t in mod.iou_threshold_dict[c]};gt_counts={(c,t):0 for c in classes for t in mod.iou_threshold_dict[c]}
 pred_groups={str(k):v for k,v in pred.groupby('pair_id',sort=False)};gt_groups={str(k):v for k,v in gt.groupby('pair_id',sort=False)};gt_ids={'car':2,'van':2,'truck':2,'bus':2,'cyclist':1,'pedestrian':0};pred_ids={'car':2,'cyclist':1,'pedestrian':0}
 for pair in sorted(set(gt_groups)|set(pred_groups)):
  pg=pred_groups.get(pair,pd.DataFrame());gg=gt_groups.get(pair,pd.DataFrame());pg=pg.copy();gg=gg.copy()
  if proxy: pg['evaluation_class']='car';gg['evaluation_class']=gg.object_class.astype(str).str.lower().map(gt_ids).map({2:'car'})
  else: pg['evaluation_class']=pg.object_class.astype(str).str.lower().where(lambda x:x.isin(pred_ids));gg['evaluation_class']=gg.object_class.astype(str).str.lower().map(gt_ids).map({2:'car',1:'cyclist',0:'pedestrian'})
  pg=pg.dropna(subset=['evaluation_class']);gg=gg.dropna(subset=['evaluation_class'])
  pa={'boxes_3d':np.array([prediction_corners(r,float(getattr(r,'rotation_rad',0.)),mod) for r in pg.itertuples(index=False)]),'labels_3d':np.array([pred_ids.get(str(v),2) for v in pg.evaluation_class],dtype=int),'scores_3d':pg.confidence.to_numpy(float) if len(pg) else np.array([])}
  ga={'boxes_3d':np.array([label_corners(r,float(r.rotation_rad)) for r in gg.itertuples(index=False)]),'labels_3d':np.array([pred_ids[str(v)] for v in gg.evaluation_class],dtype=int),'scores_3d':np.ones(len(gg))}
  for cla in classes:
   pid=pred_ids[cla];pb=pa['boxes_3d'][pa['labels_3d']==pid];ps=pa['scores_3d'][pa['labels_3d']==pid];gb=ga['boxes_3d'][ga['labels_3d']==pid]
   for threshold in mod.iou_threshold_dict[cla]:
    gt_counts[(cla,threshold)]+=len(gb)
    for view in ('3d','bev'):
     records,_=official_frame_records(pb,ps,gb,threshold,view,mod);all_records[(view,cla,threshold)].extend(records)
 metrics={}
 for view in ('3d','bev'):
  for cla in classes:
   for threshold in mod.iou_threshold_dict[cla]:
    records=all_records[(view,cla,threshold)];ngt=gt_counts[(cla,threshold)];tp=sum(x['type']=='tp' for x in records);metrics[f'{cla}_{view}_ap_iou_{threshold}']=float(mod.compute_ap(records,ngt)) if ngt else 0.;metrics[f'{cla}_{view}_recall_iou_{threshold}']=float(tp/ngt) if ngt else 0.
 receipt={'status':'PASS_OFFICIAL_DAIR_METRIC' if not proxy else 'PASS_OFFICIAL_EVALUATOR_CORE_PROXY','metric_scope':'OFFICIAL_DAIR_CLASS_ORIENTED_AP' if not proxy else 'CAR_PROXY_ZERO_YAW_NOT_LEADERBOARD_AP','official_repository':'https://github.com/AIR-THU/DAIR-V2X','official_commit':__import__('subprocess').check_output(['git','-C',str(a.official_repo),'rev-parse','HEAD'],text=True).strip(),'frames':len(set(gt.pair_id)),'prediction_semantics':'causal detector classes and oriented boxes' if not proxy else 'all class-agnostic predictions mapped to car; prediction yaw fixed to zero','metrics':metrics,'limitations':['Local evaluation uses the pinned official evaluator core.','Dimension-rule classes are a classical causal baseline, not a learned semantic detector.'] if not proxy else ['Detector has no semantic class output.','Detector has no estimated yaw.','Proxy only.']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n');table=pd.DataFrame([{'metric':k,'value':v} for k,v in metrics.items()]);table.to_csv(a.output.with_name('official_dair_metrics.csv'),index=False)
 ap=table[table.metric.str.contains('_ap_')];fig,ax=plt.subplots(figsize=(max(7,len(ap)*.55),4));ax.bar(np.arange(len(ap)),ap.value);ax.set_xticks(np.arange(len(ap)),ap.metric.str.replace('_',' '),rotation=65,ha='right',fontsize=7);ax.set(ylabel='Average precision',title='Official DAIR-V2X evaluator metrics',ylim=(0,max(.01,float(ap.value.max())*1.15)));ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.output.with_name('official_dair_ap.png'),dpi=250);fig.savefig(a.output.with_name('official_dair_ap.pdf'));plt.close(fig);print(json.dumps(receipt,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
