"""Independent materialized-graph replays and OOF/no-gold inference audit."""
from __future__ import annotations
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .experiment import write_json
from .sequential_stage12 import ROOT,load_study
from .sequential_retriever import prepare_contexts,model_scores,predicted_ranking
from .sequential_infer import infer


def materialize(context,deleted,key):
    edges=context['edges']-set(map(tuple,deleted))
    adj=[[] for _ in context['adjacency']];incoming=[[] for _ in context['incoming']]
    for u,v in sorted(edges): adj[u].append(v);incoming[v].append(u)
    return prepare_contexts({'audit':{'name':key,'edges':edges,'adjacency':adj,'incoming':incoming}})['audit']


def main():
    config,data,contexts,engine=load_study()
    lock=json.loads((ROOT/'test_lock.json').read_text())
    for filename,digest in lock['selection_hashes'].items():assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest
    for filename,digest in lock['code_hashes'].items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    original=json.loads(Path('ARTIFACTS.json').read_text())
    for filename,record in original['files'].items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==record['sha256']
    for fold in range(config['oof_folds']):
        audit=json.loads((ROOT/f'oof{fold}_audit.json').read_text())
        assert set(audit['fitting_groups']).isdisjoint(audit['heldout_groups'])
        assert set(audit['fitting_qids']).isdisjoint(audit['heldout_qids'])
        fold_contexts=pickle.loads((ROOT/f'oof{fold}_contexts.pkl').read_bytes())
        hidden=set(audit['hidden_sources'])
        assert all(not hidden.intersection(u for u,v in c['edges']) for c in fold_contexts.values())
        cases=pickle.loads((ROOT/f'oof{fold}_intervention_cases.pkl').read_bytes())
        assert set(c['qid'] for c in cases)==set(audit['heldout_qids'])
        assert all(c['fold']==fold for c in cases)
    variant=json.loads((ROOT/'stage1_selection.json').read_text())['variant']
    ranker=pickle.loads((ROOT/'ranker.pkl').read_bytes())
    policies=json.loads((ROOT/'stage3_selection.json').read_text())['policies']
    models=pickle.loads((ROOT/'utility_models.pkl').read_bytes())
    qmap={q['qid']:q for q in data['queries']}
    rng=np.random.default_rng(611)
    replays=0;inference_replays=0;times=[]
    for condition,context in contexts.items():
        cases=pickle.loads((ROOT/f'test_{condition}_cases.pkl').read_bytes())
        selected=rng.choice(len(cases),size=6,replace=False)
        records={}
        wanted={cases[int(i)]['qid'] for i in selected}
        with gzip.open(ROOT/f'test_{condition}_rankings.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['qid'] in wanted:records[r['qid']]=r
        for case_index in selected:
            case=cases[int(case_index)];query=qmap[case['qid']]
            for edge_index in sorted({0,len(case['edges'])//2,len(case['edges'])-1}):
                if edge_index<0 or not case['edges']:continue
                edge=case['edges'][edge_index]
                actual_context=materialize(context,[edge],f'audit/{condition}/{query["qid"]}/{edge_index}')
                r=engine.retrieve(query,actual_context,variant)
                actual=predicted_ranking(r,model_scores(ranker,r.features))[:40]
                np.testing.assert_array_equal(actual,case['single_rankings'][edge_index]);replays+=1
            record=records[query['qid']]
            for name,policy in policies.items():
                actual=infer(engine,query,context,condition,variant,ranker,policy,models[policy['model']])
                np.testing.assert_array_equal(actual['ranking'][:40],record['rankings'][name])
                assert [list(e) for e in actual['actions']]==record['actions'][name]
                materialized=materialize(context,actual['actions'],f'audit/{condition}/{query["qid"]}/{name}')
                r=engine.retrieve(query,materialized,variant)
                np.testing.assert_array_equal(predicted_ranking(r,model_scores(ranker,r.features))[:40],actual['ranking'][:40])
                inference_replays+=1
            # Indexed formal-query timing: excludes model/index loading and label/oracle calculation.
            engine.query_cache.pop(query['qid'],None)
            baseline=infer(engine,query,context,condition,variant,ranker)
            engine.query_cache.pop(query['qid'],None)
            with_policy=infer(engine,query,context,condition,variant,ranker,policies['utility3'],models[policies['utility3']['model']])
            times.append({'qid':query['qid'],'condition':condition,'ranker_seconds':baseline['seconds'],'with_utility3_seconds':with_policy['seconds']})
            engine.trim_cache()
    pd.DataFrame(times).to_csv(ROOT/'inference_timings.csv',index=False)
    write_json(ROOT/'replay_audit.json',{'all_passed':True,'original_files_unchanged':len(original['files']),
        'OOF_fold_audits_passed':config['oof_folds'],'single_deletion_materialized_replays':replays,
        'production_inference_and_materialized_policy_replays':inference_replays,'timing_queries':len(times),
        'timing_note':'stored formal-query vectors; excludes index/model loading and all offline intervention-label calculations'})
    print((ROOT/'replay_audit.json').read_text())


if __name__=='__main__':main()
