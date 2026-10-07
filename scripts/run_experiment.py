#!/usr/bin/env python3
"""Run the proposed object-mode scheduler and all prespecified comparators."""

from __future__ import annotations

import argparse, hashlib, json, math, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml


def unit(*parts: object) -> float:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") / (2**64-1)


def greedy_match(pred: list[dict[str, Any]], gt: pd.DataFrame, radius: float) -> dict[str, bool]:
    result = {str(v["prediction_id"]): False for v in pred}
    if not pred or gt.empty: return result
    points = gt[["x_m","y_m","z_m"]].to_numpy(float); used: set[int] = set()
    for value in sorted(pred, key=lambda v: -float(v["confidence"])):
        p=np.array([value["x_m"],value["y_m"],value["z_m"]]); distances=np.linalg.norm(points-p,axis=1)
        for i in np.argsort(distances):
            if int(i) not in used and distances[i] <= radius: used.add(int(i)); result[str(value["prediction_id"])]=True; break
    return result


def official_match(received: list[tuple[dict[str,Any],float]], candidates: dict[str,list[tuple[str,float]]], gt_ids: list[str], threshold: float=.3) -> set[str]:
    """Pinned DAIR ground-truth-order, best-IoU one-to-one matching."""
    available={str(row['prediction_id']) for row,_ in received if str(row.get('object_class','')).lower()=='car'};used=set();matched=set()
    for gid in map(str,gt_ids):
        options=[(iou,pid) for pid,iou in candidates.get(gid,[]) if pid in available and pid not in used and iou>=threshold]
        if options:
            _,pid=max(options,key=lambda x:(x[0],x[1]));used.add(pid);matched.add(pid)
    return matched


def official_ap(records: list[tuple[float,bool]], num_gt: int) -> float:
    if not records or num_gt<=0:return 0.
    ordered=sorted(records,key=lambda x:-x[0]);tp=np.cumsum([int(v[1]) for v in ordered]);precision=tp/np.arange(1,len(tp)+1);recall=tp/num_gt
    precision=np.maximum.accumulate(precision[::-1])[::-1];idx=np.where(recall[1:]!=recall[:-1])[0];return float(np.sum((recall[idx+1]-recall[idx])*precision[idx+1]))


MODES = (("summary", .22, .72), ("quantized", .48, .91), ("full", 1.0, 1.0))


def select(rows: list[dict[str,Any]], controller: str, budget: float, base_latency: float,
           loss: float, cbr: float, capacity_bytes: float, seed: str) -> list[tuple[dict[str,Any],str,int,float]]:
    adaptive = controller in {"safeslice_adaptive_payload","safeslice_paper_spec_projection","lyapunov_dpp","where2comm_object_proxy","codefilling_object_proxy","proposed_no_radio","proposed_no_fairness","proposed"}
    candidates=[]
    for row in rows:
        modes = MODES if adaptive else (("full",1.0,1.0),)
        for mode, scale, quality in modes:
            size=max(48,int(row["payload_bytes"]*scale)); fresh=max(0.,1-float(row["information_age_ms"])/float(row["deadline_ms"]))
            slack=float(row["deadline_ms"])-base_latency-size/max(capacity_bytes,1)*100.
            pdeadline=(1-loss)/(1+math.exp(-max(-40,min(40,slack/8))))
            risk=float(row["risk_score"]); conf=float(row["confidence"]); occ=.5+.5*float(row["occlusion_probability"])
            if controller=="proportional_fair_fifo": priority=-float(row["capture_timestamp_us"])
            elif controller in {"earliest_deadline_first","safeslice_style_projection","safeslice_adaptive_payload","safeslice_paper_spec_projection"}: priority=-float(row["deadline_ms"])+float(row["information_age_ms"])+1e-6*quality
            elif controller=="lyapunov_dpp":
                priority=risk*conf*occ*fresh*quality-.35*size/max(capacity_bytes,1.)
            elif controller=="where2comm_object_proxy":
                priority=conf*fresh*quality/size
            elif controller=="codefilling_object_proxy":
                grid=(round(float(row['x_m'])/5),round(float(row['y_m'])/5));priority=(conf*fresh*quality/size)+1e-3*unit(seed,*grid)
            else:
                radio = 1.0 if controller=="proposed_no_radio" else pdeadline
                priority=risk*conf*occ*fresh*radio*quality/size
            candidates.append((priority,row,mode,size,quality,pdeadline))
    candidates.sort(key=lambda v:(-v[0],str(v[1]["prediction_id"]),-v[4]))
    chosen=[]; used=0.; object_ids=set()
    for _,row,mode,size,quality,pdeadline in candidates:
        pid=str(row["prediction_id"])
        if pid in object_ids or used+size>budget: continue
        # Feasibility-first admission for RF variants.
        if controller.startswith("proposed") and pdeadline < .15: continue
        # Paper-specification SafeSlice adapter: nearest payload mode admitted
        # only when the analytical cost predictor satisfies the latency-risk
        # contract. This is not labelled official code (upstream is empty).
        if controller=="safeslice_paper_spec_projection" and pdeadline < .90: continue
        chosen.append((row,mode,size,quality)); object_ids.add(pid); used+=size
    return chosen


def run(sequence: str, controller: str, load: int, repetition: int, pairs: pd.DataFrame,
        predictions: dict[str,list[dict]], match_maps: dict[str,dict[str,bool]], truth_counts: dict[str,int],
        official_candidates: dict[str,dict[str,list[tuple[str,float]]]], official_gt_ids: dict[str,list[str]], traces: list[pd.DataFrame], cfg: dict) -> dict:
    c=cfg["campaign"]; trace=traces[int(unit(sequence,load,repetition)*len(traces))%len(traces)]; offset=int(unit(sequence,load,repetition,"o")*len(trace))
    debt=0.; critical=critical_ok=failures=matched=truth_count=0; lat=[]; prbs=[]; fair=[]; bgshares=[]; bg_sla=[]; utility=0.; oracle_matches=0; decision_ns=[];decision_cpu_ns=[];decision_objects=0;official_records=[];official_tp=official_truth=0;official_risk_utility=0.
    for fi,pair in enumerate(pairs.itertuples(index=False)):
        net=trace.iloc[(offset+fi)%len(trace)]; capacity=max(1.,float(net.capacity_bps)*float(c["frame_interval_ms"])/8000); base=float(net.latency_ms); loss=float(net.packet_loss); cbr=float(net.channel_busy_pct)
        rows=predictions.get(str(pair.pair_id),[]); matches=match_maps.get(str(pair.pair_id),{}); oracle_matches+=sum(matches.values()); truth_count+=truth_counts.get(str(pair.pair_id),0)
        background_offer=load*capacity*.30
        if controller in {"proposed","lyapunov_dpp","where2comm_object_proxy","codefilling_object_proxy"}: max_share=max(.10,1-float(c["fairness_floor"])-debt)
        elif controller=="proposed_no_fairness": max_share=.90
        elif controller in {"safeslice_style_projection","safeslice_adaptive_payload","safeslice_paper_spec_projection"}: max_share=.80
        else: max_share=.50
        target_budget=capacity*max_share
        started=time.perf_counter_ns();cpu_started=time.process_time_ns();chosen=select(rows,controller,target_budget,base,loss,cbr,capacity,f"{sequence}:{load}:{repetition}:{fi}");decision_cpu_ns.append(time.process_time_ns()-cpu_started);decision_ns.append(time.perf_counter_ns()-started);decision_objects+=len(rows)
        demand=sum(v[2] for v in chosen); target_service=min(demand,target_budget); bg_service=min(background_offer,max(0.,capacity-target_service)); total=max(target_service+bg_service,1e-9); bgshare=bg_service/total
        required_background=min(background_offer,float(c["fairness_floor"])*capacity);raw_bg_sla=bg_service/max(required_background,1e-9)
        if controller in {"proposed","lyapunov_dpp","where2comm_object_proxy","codefilling_object_proxy"}: debt=min(.45,max(0.,debt+.12*(1.-raw_bg_sla)))
        # The safety denominator is every generated critical object, including
        # rejected updates. Selection must never erase a deadline failure.
        source_critical_ids={str(value["prediction_id"]) for value in rows if float(value["risk_score"])>=float(c["critical_risk_threshold"])}
        cursor=0.; frame_critical=len(source_critical_ids); frame_critical_ok=0;received_rows=[]
        for row,mode,size,quality in chosen:
            cursor+=size; delay=base+cursor/max(target_service,1)*float(c["frame_interval_ms"]); received=unit(sequence,load,repetition,fi,row["prediction_id"])>=loss and delay<=float(c["maximum_delay_ms"])
            iscritical=str(row["prediction_id"]) in source_critical_ids
            if received:
                lat.append(delay); utility+=quality*int(matches.get(str(row["prediction_id"]),False)); matched+=int(matches.get(str(row["prediction_id"]),False))
                received_rows.append((row,quality))
                if iscritical and delay<=float(row["deadline_ms"]): frame_critical_ok+=1
        pair_id=str(pair.pair_id);frame_gt=official_gt_ids.get(pair_id,[]);frame_tp=official_match(received_rows,official_candidates.get(pair_id,{}),frame_gt);official_truth+=len(frame_gt);official_tp+=len(frame_tp)
        for row,quality in received_rows:
            if str(row.get('object_class','')).lower()=='car':
                pid=str(row['prediction_id']);is_tp=pid in frame_tp;official_records.append((float(row['confidence'])*quality,is_tp));official_risk_utility+=float(row['risk_score'])*quality*int(is_tp)
        critical+=frame_critical; critical_ok+=frame_critical_ok; failures+=int(frame_critical>frame_critical_ok)
        bgshares.append(bgshare);bg_sla.append(min(1.,raw_bg_sla));services=[target_service,*([bg_service/load]*load)]; den=len(services)*sum(x*x for x in services); fair.append(sum(services)**2/den if den else 1.)
        prbs.append(106*target_service/max(capacity,1))
    n=max(len(pairs),1)
    return {"run_id":hashlib.sha256(f"{sequence}:{controller}:{load}:{repetition}".encode()).hexdigest()[:20],"sequence_id":sequence,"controller_id":controller,"background_streams":load,"repetition":repetition,"trace_id":str(trace.iloc[0].trace_id),"frames":len(pairs),
            "critical_on_time_ratio":critical_ok/max(critical,1),"contract_failure_rate":failures/n,"perception_recall":matched/max(truth_count,1),"perception_utility":utility/max(truth_count,1),"oracle_full_recall":oracle_matches/max(truth_count,1),
            "official_car_3d_recall_03":official_tp/max(official_truth,1),"official_car_3d_ap_03":official_ap(official_records,official_truth),"official_risk_weighted_utility":official_risk_utility/max(official_truth,1),
            "p95_latency_ms":float(np.quantile(lat,.95)) if lat else math.nan,"mean_target_prbs":float(np.mean(prbs)),"mean_fairness":float(np.mean(fair)),"background_service_share":float(np.mean(bgshares)),"background_sla_ratio":float(np.mean(bg_sla)),"fairness_pass_fraction":float(np.mean(np.array(bg_sla)>=1.-1e-12)),
            "decision_time_ms_mean":float(np.mean(decision_ns)/1e6),"decision_time_ms_p95":float(np.quantile(decision_ns,.95)/1e6),"decision_time_ms_p99":float(np.quantile(decision_ns,.99)/1e6),"decision_time_ms_max":float(np.max(decision_ns)/1e6),"decision_time_us_per_candidate":float(sum(decision_ns)/1e3/max(decision_objects,1)),
            "decision_cpu_time_ms_mean":float(np.mean(decision_cpu_ns)/1e6),"decision_cpu_time_ms_p99":float(np.quantile(decision_cpu_ns,.99)/1e6),"decision_cpu_time_ms_max":float(np.max(decision_cpu_ns)/1e6),"decision_count":len(decision_ns),
            "network_provenance":"MEASURED_SEE_V2X_CROSSFIT_CAPACITY_REPLAY","application_provenance":"MEASURED_ALIGNED_PERCEPTION_OUTPUT","physical_radio_claim":False}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--config",type=Path,required=True);ap.add_argument("--resume",action="store_true");args=ap.parse_args();config=yaml.safe_load(args.config.read_text());base=args.config.resolve().parents[1]; path=lambda k:(base/config["paths"][k]).resolve()
    pairs=pd.read_parquet(path("dataset_dir")/"cooperative_pairs.parquet");pred=pd.read_parquet(path("aligned_predictions"));gt=pd.read_parquet(path("dataset_dir")/"ground_truth_objects.parquet");tr=pd.read_parquet(path("measured_traces"))
    allowed={"LEAVE_SCENARIO_OUT_CBR_P90_ENVELOPE","LOSO_SCI_MCS_PRB_Q90_GBR_V1","LOSO_SCI_MCS_PRB_Q10_CQR_V2"}
    if not set(tr.capacity_model).issubset(allowed):raise SystemExit("capacity evidence gate failed")
    predmap={str(k):v.assign(payload_bytes=v.astype(str).agg('|'.join,axis=1).str.len()+96).to_dict('records') for k,v in pred.groupby('pair_id',sort=False)};gtmap={str(k):v for k,v in gt.groupby('pair_id',sort=False)};traces=[v.reset_index(drop=True) for _,v in tr.groupby('trace_id')]
    cand=pd.read_parquet(path('official_candidates'));official_candidates={}
    for (pair,gid),v in cand.groupby(['pair_id','ground_truth_id'],sort=False):official_candidates.setdefault(str(pair),{})[str(gid)]=list(zip(v.prediction_id.astype(str),v.iou_3d.astype(float)))
    car_gt=gt[gt.object_class.astype(str).str.lower().isin(['car','van','truck','bus'])];official_gt_ids={str(k):v.ground_truth_id.astype(str).tolist() for k,v in car_gt.groupby('pair_id',sort=False)}
    match_maps={pair_id:greedy_match(values,gtmap.get(pair_id,pd.DataFrame()),float(config['campaign']['match_radius_m'])) for pair_id,values in predmap.items()};truth_counts={pair_id:len(value) for pair_id,value in gtmap.items()};oracle=sum(sum(v.values()) for v in match_maps.values())/max(sum(truth_counts.values()),1)
    if oracle < .05: raise SystemExit(f"PERCEPTION ORACLE GATE FAILED: aligned full-communication recall={oracle:.6f} < 0.05")
    out=path("output_dir");out.mkdir(parents=True,exist_ok=True);result=out/"run_summaries.parquet";old=pd.read_parquet(result) if args.resume and result.exists() else pd.DataFrame();rows=old.to_dict('records') if len(old) else [];done=set(old.run_id) if len(old) else set();seqs=sorted(pairs.source_sequence_id.astype(str).unique());total=len(seqs)*len(config['campaign']['controllers'])*len(config['campaign']['background_streams'])*int(config['campaign']['repetitions'])
    for seq in seqs:
        ps=pairs[pairs.source_sequence_id.astype(str)==seq].sort_values('vehicle_pointcloud_timestamp_us')
        for load in config['campaign']['background_streams']:
            for rep in range(1,int(config['campaign']['repetitions'])+1):
                for ctl in config['campaign']['controllers']:
                    rid=hashlib.sha256(f"{seq}:{ctl}:{load}:{rep}".encode()).hexdigest()[:20]
                    if rid not in done:rows.append(run(seq,ctl,int(load),rep,ps,predmap,match_maps,truth_counts,official_candidates,official_gt_ids,traces,config));done.add(rid)
        pd.DataFrame(rows).to_parquet(result,index=False);print(f"checkpoint runs={len(rows)}/{total} sequence={seq}",flush=True)
    frame=pd.DataFrame(rows);manifest={"status":"COMPLETE","runs":len(frame),"sequences":len(seqs),"oracle_full_recall":oracle,"output":str(result),"physical_radio_claim":False};(out/'campaign_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
