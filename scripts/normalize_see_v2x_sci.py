#!/usr/bin/env python3
"""Normalize official SEE-V2X sidelink-control-information measurements."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); rows=[]; sources=[]
    for path in sorted(a.input_dir.rglob('sci.csv')):
        d=pd.read_csv(path,low_memory=False); required={'rx_timestamp_us','mcs_idx','nof_prb_pssch','retransmission'}
        if not required.issubset(d.columns): continue
        rel=str(path.relative_to(a.input_dir)); scenario=re.search(r'emulating_applications/([^/]+/[^/]+/intersection_[^/]+)',rel)
        if not scenario: continue
        out=pd.DataFrame({k:pd.to_numeric(d[k],errors='coerce') for k in required})
        for optional in ('sf_idx','pssch_start_idx','N_x_id','resource_reserv'):
            if optional in d: out[optional]=pd.to_numeric(d[optional],errors='coerce')
        if 'resource_reserv' in d: out['resource_reserv']=pd.to_numeric(d.resource_reserv,errors='coerce')
        out=out.dropna(subset=list(required));out['scenario_id']=scenario.group(1);out['source_file']=rel;out['source_sha256']=sha(path);rows.append(out);sources.append({'path':rel,'sha256':sha(path),'rows':len(out)})
    if not rows: raise SystemExit('SCI GATE FAILED: no official SEE-V2X sci.csv files found')
    result=pd.concat(rows,ignore_index=True)
    if result.scenario_id.nunique()<8: raise SystemExit(f'SCI GATE FAILED: require >=8 independent scenarios; found {result.scenario_id.nunique()}')
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(a.output,index=False)
    receipt={'status':'PASS_MEASURED_SCI_GATE','rows':len(result),'scenarios':int(result.scenario_id.nunique()),'mcs_values':sorted(map(int,result.mcs_idx.unique())),'prb_range':[float(result.nof_prb_pssch.min()),float(result.nof_prb_pssch.max())],'output':str(a.output.resolve()),'output_sha256':sha(a.output),'sources':sources,'provenance':'OFFICIAL_SEE_V2X_MEASURED_SCI'}
    a.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
