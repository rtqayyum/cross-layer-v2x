#!/usr/bin/env python3
"""Precompute sparse official DAIR car-IoU candidates for controller replay."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_v2x_official_metric import label_corners,prediction_corners,load_official

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--ground-truth',type=Path,required=True);p.add_argument('--official-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();pred=pd.read_parquet(a.predictions);gt=pd.read_parquet(a.ground_truth);mod=load_official(a.official_repo)
 pred=pred[pred.object_class.astype(str).str.lower()=='car'];gt=gt[gt.object_class.astype(str).str.lower().isin(['car','van','truck','bus'])];pg={str(k):v for k,v in pred.groupby('pair_id',sort=False)};rows=[]
 for pair,gg in gt.groupby('pair_id',sort=False):
  pp=pg.get(str(pair));
  if pp is None:continue
  pboxes=[prediction_corners(r,float(r.rotation_rad),mod) for r in pp.itertuples(index=False)];gboxes=[label_corners(r,float(r.rotation_rad)) for r in gg.itertuples(index=False)]
  pmeta=list(pp[['prediction_id','confidence']].itertuples(index=False,name=None));gids=gg.ground_truth_id.astype(str).tolist()
  for gi,gb in enumerate(gboxes):
   gc=gb.mean(0);gr=np.max(np.linalg.norm(gb[:4,:2]-gc[:2],axis=1))
   for pi,pb in enumerate(pboxes):
    pc=pb.mean(0);pr=np.max(np.linalg.norm(pb[:4,:2]-pc[:2],axis=1))
    if np.linalg.norm(gc[:2]-pc[:2])>gr+pr or min(gb[:,2].max(),pb[:,2].max())<=max(gb[:,2].min(),pb[:,2].min()):continue
    try:i3,ib=mod.box3d_iou(gb[mod.perm_label],pb[mod.perm_pred])
    except Exception:i3,ib=0.,0.
    if i3>0 or ib>0:rows.append({'pair_id':str(pair),'ground_truth_id':gids[gi],'prediction_id':str(pmeta[pi][0]),'confidence':float(pmeta[pi][1]),'iou_3d':float(i3),'iou_bev':float(ib)})
 result=pd.DataFrame(rows,columns=['pair_id','ground_truth_id','prediction_id','confidence','iou_3d','iou_bev']);a.output.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(a.output,index=False);receipt={'status':'PASS_OFFICIAL_IOU_CANDIDATES','pairs':int(result.pair_id.nunique()),'candidates':len(result),'predictions_sha256':sha(a.predictions),'ground_truth_sha256':sha(a.ground_truth),'output_sha256':sha(a.output),'official_commit':__import__('subprocess').check_output(['git','-C',str(a.official_repo),'rev-parse','HEAD'],text=True).strip()};a.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
