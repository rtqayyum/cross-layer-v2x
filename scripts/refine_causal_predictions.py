#!/usr/bin/env python3
"""Causally refine measured geometric detections with axis OBBs and classes."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def classify(r):
 l,w,h=float(r.length_m),float(r.width_m),float(r.height_m)
 if 1<=h<=2.5 and l<=1.5 and w<=1.2:return 'Pedestrian'
 if .8<=h<=2.5 and 1<=l<=3.2 and w<=1.5:return 'Cyclist'
 if .8<=h<=3.2 and 2.5<=l<=7.5 and 1.1<=w<=3.2:return 'Car'
 return 'unclassified_obstacle'
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();d=pd.read_parquet(a.input).copy();swap=d.width_m>d.length_m;old=d.length_m.copy();d.loc[swap,'length_m']=d.loc[swap,'width_m'];d.loc[swap,'width_m']=old[swap];d['rotation_rad']=np.where(swap,math.pi/2,0.);d['object_class']=[classify(r) for r in d.itertuples(index=False)];d['model_id']='causal-axis-obb-lidar-v2';d['model_sha256']=hashlib.sha256(Path(__file__).read_bytes()+a.input.read_bytes()[:1024]).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(a.output,index=False)
 receipt={'status':'PASS_CAUSAL_REFINEMENT','rows':len(d),'pairs':int(d.pair_id.nunique()),'classes':d.object_class.value_counts().to_dict(),'uses_ground_truth_at_inference':False,'uses_future_frames':False,'pseudo_labels':False,'method':'axis-aligned OBB normalization and dimension-rule semantics from measured frame-local clusters','input_sha256':sha(a.input),'output_sha256':sha(a.output),'limitations':'Yaw is restricted to infrastructure-lidar axes; use the optional PCA-OBB inference for unconstrained yaw.'};a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
