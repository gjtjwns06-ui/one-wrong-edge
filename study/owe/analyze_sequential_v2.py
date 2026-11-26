"""Paired stage comparisons and additional edge-ML value on reused test data."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from .analyze import cluster_interval
from .experiment import write_json
from .sequential_stage12 import ROOT, load_study


def condition_frames(frame, keys):
    result={name:sub.copy() for name,sub in frame.groupby("condition")}
    for name,conditions in [("random20_mean",["random20_s17","random20_s29","random20_s43"]),
                            ("random20_unseen_mean",["random20_s29","random20_s43"])]:
        sub=frame[frame.condition.isin(conditions)]
        numeric=list(sub.select_dtypes(include="number").columns.difference(keys))
        pooled=sub.groupby(keys,as_index=False)[numeric].mean()
        pooled["condition"]=name
        result[name]=pooled
    return result


def main():
    if not (ROOT/"run_complete.json").exists(): raise RuntimeError("Complete the frozen test evaluation first")
    config,data,contexts,engine=load_study()
    results=pd.read_csv(ROOT/"results.csv")
    # Form event indicators before averaging repeated corruption seeds.
    results['acted_event']=(results.deletions>0).astype(float)
    results['improved_event']=(results.delta_vs_stage2>1e-8).astype(float)
    results['harmed_event']=(results.delta_vs_stage2< -1e-8).astype(float)
    assert not results.duplicated(["condition","qid","method","budget"]).any()
    frames=condition_frames(results,["qid","group","topic","method","budget"])
    summary=[]
    for condition,frame in frames.items():
        for (method,budget),sub in frame.groupby(["method","budget"]):
            lo,hi=cluster_interval(sub,"delta_vs_stage2")
            summary.append({"condition":condition,"method":method,"budget":int(budget),
                "queries":sub.qid.nunique(),"source_blocks":sub.group.nunique(),
                "recall":sub.recall.mean(),"ndcg":sub.ndcg.mean(),"complete":sub.complete.mean(),
                "delta_vs_stage2":sub.delta_vs_stage2.mean(),"ci_low":lo,"ci_high":hi,
                "improved":sub.improved_event.mean(),"harmed":sub.harmed_event.mean(),
                "mean_deletions":sub.deletions.mean(),"action_rate":sub.acted_event.mean(),
                "wrong_deletions":sub.wrong_deletions.sum(),"all_deletions":sub.deletions.sum(),
                "wrong_deletion_precision":sub.wrong_deletions.sum()/sub.deletions.sum() if sub.deletions.sum() else np.nan,
                "returned":sub.returned.mean(),"formal_tokens":sub.formal_tokens.mean(),
                "candidate_coverage":sub.candidate_coverage.mean(),"candidate_ceiling":sub.candidate_ceiling.mean(),
                "transitions":sub.transitions.mean()})
    summary=pd.DataFrame(summary);summary.to_csv(ROOT/"summary.csv",index=False)
    pair_names=[("stage2_proof","v1_local"),("stage2_proof","stage1_text"),
                ("stage2_proof","hand_mix"),("utility3","stage2_proof"),
                ("wrongness3","stage2_proof"),("utility3","matched_utility3"),
                ("wrongness3","matched_wrongness3"),("utility3","source_verified")]
    pair_names.extend([('wrongness_filter','stage2_proof'),('wrongness_filter','matched_wrongness_filter')])
    pairs=[]
    for condition,frame in frames.items():
        sub=frame[frame.budget==20]
        for first,second in pair_names:
            joined=sub[sub.method==first][["qid","group","recall"]].merge(
                sub[sub.method==second][["qid","recall"]],on="qid",suffixes=("_a","_b"))
            joined["difference"]=joined.recall_a-joined.recall_b
            lo,hi=cluster_interval(joined,"difference")
            pairs.append({"condition":condition,"comparison":first+" minus "+second,
                          "difference":joined.difference.mean(),"ci_low":lo,"ci_high":hi})
    pd.DataFrame(pairs).to_csv(ROOT/"comparisons.csv",index=False)
    # Same candidate cap control and original candidate coverage are separate from ranking.
    qmap={q["qid"]:q for q in data["queries"]}
    clean20=results[(results.condition=='clean')&(results.budget==20)]
    base_changes=clean20[clean20.method=='stage2_proof'][['qid','group','topic','recall']].merge(
        clean20[clean20.method=='v1_local'][['qid','recall']],on='qid',suffixes=('_new','_old'))
    base_changes['difference']=base_changes.recall_new-base_changes.recall_old
    base_changes.to_csv(ROOT/'base_ranker_query_changes.csv',index=False)
    write_json(ROOT/'base_ranker_change_counts.json',{
        'queries':len(base_changes),'improved':int((base_changes.difference>1e-8).sum()),
        'harmed':int((base_changes.difference< -1e-8).sum()),'unchanged':int((base_changes.difference.abs()<=1e-8).sum())})
    old=pd.read_csv("runs/retriever_diagnostic_v1/stage_decomposition.csv")
    coverage=[]
    lexical_coverage={}
    for q in [q for q in data["queries"] if q["split"]=="test"]:
        info=engine.query_info(q)
        lexical_coverage[q["qid"]]=len(set(info["lexical"][:256])&set(q["gold"]))/len(q["gold"])
        engine.trim_cache()
    base=results[(results.method=="stage2_proof")&(results.budget==20)]
    joined=base.merge(old[["qid","condition","two_hop_gold_coverage"]],on=["qid","condition"])
    joined["lexical256_coverage"]=joined.qid.map(lexical_coverage)
    cov_frames=condition_frames(joined[["qid","group","topic","condition","candidate_coverage","candidate_ceiling","two_hop_gold_coverage","lexical256_coverage"]],["qid","group","topic"])
    for condition,sub in cov_frames.items():
        coverage.append({"condition":condition,"old_pool_coverage":sub.two_hop_gold_coverage.mean(),
            "new_pool_coverage":sub.candidate_coverage.mean(),"new_pool_ceiling20":sub.candidate_ceiling.mean(),
            "lexical256_coverage":sub.lexical256_coverage.mean()})
    pd.DataFrame(coverage).to_csv(ROOT/"candidate_coverage_summary.csv",index=False)
    effects=pd.concat([pd.read_csv(ROOT/f"test_{name}_effects.csv.gz") for name in contexts],ignore_index=True)
    effects["wrong"]=1-effects.source_supported
    effects["positive"]=effects.benefit20>1e-8
    effects["negative"]=effects.benefit20< -1e-8
    effects.groupby(["condition","source_supported"])[["positive","negative"]].agg(["sum","mean","count"]).to_csv(ROOT/"effect_signs.csv")
    classification=[]
    for name,sub in effects.groupby("condition"):
        unique=sub.groupby(["source","target"],as_index=False).agg(wrong=("wrong","first"),prediction=("wrongness_prediction","mean"),queries=("qid","nunique"))
        if unique.wrong.nunique()==2:
            classification.append({"condition":name,"unique_edges":len(unique),"wrong_edges":int(unique.wrong.sum()),
                "ROC_AUC":roc_auc_score(unique.wrong,unique.prediction),"average_precision":average_precision_score(unique.wrong,unique.prediction),
                "note":"unique-edge mean prediction; no independent-edge significance claim"})
    pd.DataFrame(classification,columns=['condition','unique_edges','wrong_edges','ROC_AUC','average_precision','note']).to_csv(ROOT/"wrongness_classification.csv",index=False)
    # Distinct failure examples from actual final actions.
    candidates=results[(results.budget==20)&results.method.isin(["utility3","wrongness3"])]
    chosen=candidates[candidates.delta_vs_stage2< -1e-8].sort_values("delta_vs_stage2").drop_duplicates("qid").head(6)
    wanted={(r.qid,r.condition) for r in chosen.itertuples()}
    named_examples=['eulerpathpr','konigsberg','usgrfs']
    regressions=base_changes[base_changes.difference< -1e-8].nsmallest(5,'difference').qid.tolist()
    wanted.update((qid,'clean') for qid in named_examples+regressions)
    records={}
    for name in contexts:
        with gzip.open(ROOT/f"test_{name}_rankings.jsonl.gz","rt",encoding="utf-8") as f:
            for line in f:
                record=json.loads(line)
                if (record["qid"],name) in wanted: records[record["qid"],name]=record
    examples=[]
    reference=set(map(tuple,data["edges"]))
    for row in chosen.itertuples():
        rec=records[row.qid,row.condition];q=qmap[row.qid];gold=set(q["gold"])
        before=set(rec["rankings"]["stage2_proof"][:20]);after=set(rec["rankings"][row.method][:20])
        examples.append({"qid":row.qid,"condition":row.condition,"policy":row.method,"delta_recall20":row.delta_vs_stage2,
            "source":data["nodes"][q["node"]]["source"],
            "actions":[{"source":data["nodes"][u]["label"],"target":data["nodes"][v]["label"],"source_supported":(u,v) in reference} for u,v in rec["actions"][row.method]],
            "lost_gold":[data["nodes"][v]["label"] for v in sorted((before-after)&gold)],
            "gained_gold":[data["nodes"][v]["label"] for v in sorted((after-before)&gold)]})
    write_json(ROOT/"failure_cases.json",examples)
    worked=[]
    for qid in dict.fromkeys(named_examples+regressions):
        q=qmap[qid];rec=records[qid,'clean'];gold=set(q['gold'])
        worked.append({'qid':qid,'kind':'named_diagnostic' if qid in named_examples else 'base_ranker_regression',
            'source':data['nodes'][q['node']]['source'],'gold_count':len(gold),
            'old_recall20':len(gold&set(rec['rankings']['v1_local'][:20]))/len(gold),
            'new_pool_coverage':len(gold&set(rec['candidates']))/len(gold),
            'proof_recall20':len(gold&set(rec['rankings']['stage2_proof'][:20]))/len(gold),
            'utility3_recall20':len(gold&set(rec['rankings']['utility3'][:20]))/len(gold),
            'lost_vs_old':[data['nodes'][v]['label'] for v in sorted((set(rec['rankings']['v1_local'][:20])-set(rec['rankings']['stage2_proof'][:20]))&gold)],
            'gained_vs_old':[data['nodes'][v]['label'] for v in sorted((set(rec['rankings']['stage2_proof'][:20])-set(rec['rankings']['v1_local'][:20]))&gold)]})
    write_json(ROOT/'worked_examples.json',worked)
    figures=ROOT/"figures";figures.mkdir(exist_ok=True)
    conditions=["clean","random20_mean","hard20_s17","missing20_s17"]
    names=["Clean","Random +20%\n3-seed mean","Lexical-hard +20%","Missing 20%"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for method,label in [("v1_local","Original local"),("stage1_text","Expanded pool + lexical"),("stage2_proof","Learned proof-use ranker"),("utility3","+ Utility deletion ≤3")]:
        sub=summary[(summary.method==method)&(summary.budget==20)].set_index("condition").loc[conditions]
        axes[0].plot(range(len(conditions)),sub.recall*100,marker="o",label=label)
    axes[0].set_xticks(range(len(conditions)),names);axes[0].set_ylabel("Direct-premise Recall@20 (%)")
    axes[0].set_title("Candidate and ranking stages");axes[0].legend(fontsize=8)
    for i,(method,label) in enumerate([("utility1","Utility ≤1"),("utility3","Utility ≤3"),("wrongness_filter","Wrongness filter"),("source_verified","Source verification")]):
        sub=summary[(summary.method==method)&(summary.budget==20)].set_index("condition").loc[conditions]
        x=np.arange(len(conditions))+(i-1.5)*0.14;color=plt.get_cmap("tab10")(i)
        axes[1].vlines(x,sub.ci_low*100,sub.ci_high*100,color=color)
        axes[1].scatter(x,sub.delta_vs_stage2*100,color=color,label=label,s=23)
    axes[1].axhline(0,color="black",linewidth=0.7);axes[1].set_xticks(range(len(conditions)),names)
    axes[1].set_ylabel("Change vs learned ranker (percentage points)")
    axes[1].set_title("Additional intervention value; 95% block intervals");axes[1].legend(fontsize=8)
    fig.suptitle("Sequential development evaluation on reused test data",fontsize=12)
    fig.tight_layout()
    for suffix in ["png","svg"]:fig.savefig(figures/f"sequential_stages.{suffix}",dpi=220)
    plt.close(fig)
    qa={"test_queries":results.qid.nunique(),"blocks":results.group.nunique(),
        "all_output_caps_respected":bool((results.returned<=results.budget).all()),
        "finite_scores":bool(np.isfinite(results[["recall","ndcg","delta_vs_stage2"]]).all().all()),
        "lexical_and_stage1_text_identical":True,
        "source_filter_restores_clean_for_additions":True,
        "action_rate_within_mean_deletion_bound":bool((summary.action_rate<=summary.mean_deletions+1e-12).all()),
        "single_oracle_never_worse_at20":bool((results[(results.method=="oracle1")&(results.budget==20)].delta_vs_stage2>=-1e-8).all())}
    for name in contexts:
        a=results[(results.condition==name)&(results.method=="lexical")].sort_values(["qid","budget"]).recall.to_numpy()
        b=results[(results.condition==name)&(results.method=="stage1_text")].sort_values(["qid","budget"]).recall.to_numpy()
        qa["lexical_and_stage1_text_identical"] &= bool(np.array_equal(a,b))
        if name.startswith(('random','hard')):
            clean=results[(results.condition=='clean')&(results.method=='stage2_proof')].sort_values(['qid','budget']).recall.to_numpy()
            verified=results[(results.condition==name)&(results.method=='source_verified')].sort_values(['qid','budget']).recall.to_numpy()
            qa['source_filter_restores_clean_for_additions'] &= bool(np.array_equal(clean,verified))
    assert all(value for key,value in qa.items() if key not in ["test_queries","blocks"])
    write_json(ROOT/"analysis_checks.json",qa)
    print(summary[(summary.budget==20)&summary.condition.isin(conditions)&summary.method.isin(["v1_local","lexical","stage2_proof","utility1","utility3","wrongness3","source_verified","oracle1"])][["condition","method","recall","delta_vs_stage2","ci_low","ci_high","mean_deletions"]].to_string(index=False))


if __name__=="__main__":main()
