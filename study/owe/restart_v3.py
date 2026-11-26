"""From-raw scoped experiment with fresh-topic test and disjoint policy training."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time
import os
import platform
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .experiment import make_context, write_json
from .prepare import stable_int
from .retrieval import metrics, Neighborhood
from .sequential_retriever import HybridEngine, model_scores, predicted_ranking, RANK_FEATURES
from .sequential_stage12 import training_arrays, fit_ranker
from .sequential_interventions import (make_case, edge_features, utility_predictions,
    choose_actions, batch_rank, candidate_edges, EDGE_FEATURES)
from .restart_support import (ProjectedEstimator, TEXT_COLUMNS, context_from_edges,
    query_context)
from .restart_workers import generate

ROOT=Path('runs/restart_v3')
CONFIG=Path('configs/restart_v3.json')


def load(path):
    return pickle.loads(Path(path).read_bytes())


def dump(path,value):
    Path(path).write_bytes(pickle.dumps(value,protocol=5))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study():
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    data=load(config['dataset']);contexts=load(ROOT/'graphs.pkl')
    vectorizer=load(ROOT/'vectorizer.pkl')
    return config,data,contexts,HybridEngine(data,vectorizer,config)


def queries_for(data,split):
    return [q for q in data['queries'] if q['split']==split]


def initialize():
    if (ROOT/'index_manifest.json').exists():raise RuntimeError('Preserve existing initialized run')
    ROOT.mkdir(parents=True,exist_ok=True)
    config=json.loads(CONFIG.read_text(encoding='utf-8'));data=load(config['dataset'])
    hidden_signatures={q['signature'] for q in data['queries'] if q['split'] in ('policy_train','validation')}
    hidden_signatures.update(n['signature'] for n in data['nodes'] if n.get('fresh_topic_core',False))
    hidden={n['id'] for n in data['nodes'] if n['signature'] in hidden_signatures}
    public=[n['id'] for n in data['nodes'] if n['id'] not in hidden]
    vectorizer=TfidfVectorizer(tokenizer=str.split,preprocessor=None,token_pattern=None,
        lowercase=False,ngram_range=(1,2),min_df=1,sublinear_tf=True,dtype=np.float32)
    matrix=vectorizer.fit_transform([data['nodes'][i]['text'] for i in public])
    matrix=vectorizer.transform([n['text'] for n in data['nodes']]).tocsr()
    dump(ROOT/'vectorizer.pkl',vectorizer)
    contexts={};records=[]
    for spec in config['conditions']:
        started=time.perf_counter()
        if spec['mode']=='mixed':
            remove=make_context(data,{**spec,'mode':'missing','rate':spec['rate']/2},matrix)
            add=make_context(data,{**spec,'mode':'hard','rate':spec['rate']/2,'seed':spec['seed']+1},matrix)
            edges=remove['edges']|(add['edges']-set(map(tuple,data['edges'])))
            context=context_from_edges(data['nodes'],edges,spec['name'],remove['audit_changes']+add['audit_changes'])
        else:
            base=make_context(data,spec,matrix)
            context=context_from_edges(data['nodes'],base['edges'],spec['name'],base['audit_changes'])
        contexts[spec['name']]=context
        reference=set(map(tuple,data['edges']))
        changes=len(reference-context['edges'])+len(context['edges']-reference)
        records.append({**spec,'reference_edges':len(reference),'observed_edges':len(context['edges']),
            'added':len(context['edges']-reference),'removed':len(reference-context['edges']),
            'realized_change_rate':changes/len(reference),
            'failed_proposals':int(len(reference)*spec['rate'])-changes,
            'seconds':time.perf_counter()-started})
        assert not hidden.intersection(u for u,v in context['edges'])
        print(f"Constructed {spec['name']}: {records[-1]['added']} added/{records[-1]['removed']} missing",flush=True)
    dump(ROOT/'graphs.pkl',contexts)
    pd.DataFrame(records).to_csv(ROOT/'graph_conditions.csv',index=False)
    write_json(ROOT/'index_manifest.json',{'protocol_sha256':sha('RESTART_V3_PROTOCOL.md'),
        'config_sha256':sha(CONFIG),'dataset_sha256':sha(config['dataset']),
        'data_manifest_sha256':sha('data/processed/restart_v3/manifest.json'),
        'vectorizer_fitted_nodes':public,'vocabulary_size':len(vectorizer.vocabulary_),
        'query_labels_or_test_proofs_used_for_index_or_idf':False})


def candidates(config,data,contexts,engine):
    if (ROOT/'stage1_selection.json').exists():raise RuntimeError('Candidate selection already locked')
    rows=[]
    for variant in config['candidate_variants']:
        for condition in config['validation_conditions']:
            for query in queries_for(data,'validation'):
                result=engine.retrieve(query,contexts[condition],variant)
                hits=len(set(result.ids)&set(query['gold']))
                lexical=engine.query_info(query)['lexical'][:config['candidate_cap']]
                rows.append({'variant':variant['name'],'condition':condition,'qid':query['qid'],
                    'group':query['group'],'coverage':hits/len(query['gold']),
                    'ceiling20':min(20,hits)/len(query['gold']),'transitions':result.transitions,
                    'lexical_coverage':len(set(lexical)&set(query['gold']))/len(query['gold'])})
                engine.trim_cache()
        print('Candidate validation '+variant['name'],flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'stage1_validation.csv',index=False)
    summary=frame.groupby('variant',as_index=False)[['coverage','ceiling20','transitions','lexical_coverage']].mean()
    summary.to_csv(ROOT/'stage1_validation_summary.csv',index=False)
    best=summary.sort_values(['ceiling20','coverage','transitions'],ascending=[False,False,True]).iloc[0]
    variant=next(v for v in config['candidate_variants'] if v['name']==best.variant)
    write_json(ROOT/'stage1_selection.json',{'variant':variant,'validation':best.to_dict(),'selected_before_rank':True})
    print(summary.to_string(index=False),flush=True)


def rank(config,data,contexts,engine):
    if (ROOT/'stage2_selection.json').exists():raise RuntimeError('Ranker already locked')
    variant=json.loads((ROOT/'stage1_selection.json').read_text())['variant']
    train=queries_for(data,'rank_train')
    x,y,w,records=training_arrays(engine,contexts,train,variant,config,'restart-v3')
    pd.DataFrame(records).to_csv(ROOT/'ranker_training_cases.csv',index=False)
    write_json(ROOT/'stage2_training_receipt.json',{'rank_qids':[q['qid'] for q in train],
        'rank_groups':sorted({q['group'] for q in train}),'sampled_pairs':len(y),'positive':int(y.sum()),
        'feature_names':RANK_FEATURES,'text_columns':TEXT_COLUMNS})
    validation=[]
    for condition in config['validation_conditions']:
        for query in queries_for(data,'validation'):
            result=engine.retrieve(query,contexts[condition],variant)
            validation.append((query,condition,result));engine.trim_cache()
    all_x=np.concatenate([r.features for q,c,r in validation])
    names=['logistic_1','boost_15','boost_31','text_boost_31'];models={};rows=[]
    for name in names:
        started=time.perf_counter()
        if name.startswith('text_'):
            model=fit_ranker('boost_31',config,x[:,TEXT_COLUMNS],y,w)
            model['estimator']=ProjectedEstimator(model['estimator'],TEXT_COLUMNS);model['name']=name
        else:model=fit_ranker(name,config,x,y,w)
        models[name]=model;scores=model_scores(model,all_x);offset=0
        for query,condition,result in validation:
            score=scores[offset:offset+len(result.ids)];offset+=len(result.ids)
            measured=metrics(predicted_ranking(result,score),set(query['gold']),20,engine.lengths)
            rows.append({'model':name,'condition':condition,'qid':query['qid'],'group':query['group'],**measured})
        print(f'Ranker {name} fit+score {time.perf_counter()-started:.1f}s',flush=True)
    for query,condition,result in validation:
        rows.append({'model':'expanded_text','condition':condition,'qid':query['qid'],'group':query['group'],
                     **metrics(predicted_ranking(result,result.features[:,0]),set(query['gold']),20,engine.lengths)})
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'stage2_validation.csv',index=False)
    summary=frame.groupby('model',as_index=False)[['recall','ndcg','complete']].mean()
    summary.to_csv(ROOT/'stage2_validation_summary.csv',index=False)
    order={n:i for i,n in enumerate(['text_boost_31','logistic_1','boost_15','boost_31'])}
    best=summary[summary.model.isin(models)].assign(complexity=lambda f:f.model.map(order)).sort_values(
        ['recall','complete','complexity'],ascending=[False,False,True]).iloc[0]
    dump(ROOT/'ranker.pkl',models[best.model]);dump(ROOT/'text_ranker.pkl',models['text_boost_31'])
    write_json(ROOT/'stage2_selection.json',{'model':best.model,'validation':best.to_dict(),
        'stage1_sha256':sha(ROOT/'stage1_selection.json'),'selected_before_policy':True})
    print(summary.to_string(index=False),flush=True)


def fit_policies(cases,reference,config):
    x=np.concatenate([c['features'] for c in cases]);y=np.concatenate([c['benefits'][:,1] for c in cases])
    wrong=np.array([edge not in reference for c in cases for edge in c['edges']],dtype=np.int8)
    w=np.concatenate([np.full(len(c['edges']),1/max(1,len(c['edges']))) for c in cases]);w*=len(w)/w.sum()
    scaler=StandardScaler().fit(x,sample_weight=w)
    models={'ridge_10':{'kind':'utility','estimator':Ridge(alpha=10).fit(scaler.transform(x),y,sample_weight=w),'scaler':scaler}}
    reg=HistGradientBoostingRegressor(max_iter=config['utility_boost_iterations'],max_leaf_nodes=15,
        learning_rate=.08,min_samples_leaf=40,l2_regularization=1,early_stopping=False,random_state=config['random_seed'])
    reg.fit(x,y,sample_weight=w);models['utility_boost']={'kind':'utility','estimator':reg,'scaler':None}
    wc=w.copy()
    for label in [0,1]:
        m=wrong==label
        assert m.any()
        wc[m]*=w.sum()/(2*w[m].sum())
    clf=HistGradientBoostingClassifier(max_iter=config['utility_boost_iterations'],max_leaf_nodes=15,
        learning_rate=.08,min_samples_leaf=40,l2_regularization=1,early_stopping=False,random_state=config['random_seed'])
    clf.fit(x,wrong,sample_weight=wc);models['wrongness_boost']={'kind':'wrongness','estimator':clf,'scaler':None}
    write_json(ROOT/'utility_training_summary.json',{'policy_cases':len(cases),'effects':len(y),
        'positive':int((y>1e-8).sum()),'negative':int((y< -1e-8).sum()),'zero':int((np.abs(y)<=1e-8).sum()),
        'wrong_candidates':int(wrong.sum()),'edge_features':EDGE_FEATURES})
    np.savez_compressed(ROOT/'utility_training.npz',features=x,benefits=y,wrongness=wrong,weights=w)
    return models


def infer(engine,query,context,variant,ranker,policy=None,models=None):
    started=time.perf_counter()
    query={k:v for k,v in query.items() if not k.startswith('gold') and k not in ('out_of_index_count',)}
    result=engine.retrieve(query,context,variant);score=model_scores(ranker,result.features)
    actions=[]
    if policy is not None:
        edges=candidate_edges(result,query,policy.get('condition_name',context['name'].split('/')[0]),score,engine.config['utility_candidate_cap'])
        features=edge_features(engine,query,context,result,score,edges)
        predictions=utility_predictions(models[policy['model']],features)
        actions=choose_actions({'edges':edges},predictions,policy['budget'],policy['threshold'])
    if actions:
        result=engine.retrieve(query,context,variant,actions);score=model_scores(ranker,result.features)
    return {'ranking':predicted_ranking(result,score)[:40],'actions':actions,'seconds':time.perf_counter()-started}


def policy(config,data,contexts,engine):
    if (ROOT/'stage3_selection.json').exists():raise RuntimeError('Policy selection already locked')
    variant=json.loads((ROOT/'stage1_selection.json').read_text())['variant'];ranker=load(ROOT/'ranker.pkl')
    training=queries_for(data,'policy_train');rankq=queries_for(data,'rank_train')
    assert set(q['group'] for q in training).isdisjoint(q['group'] for q in rankq)
    write_json(ROOT/'stage3_training_receipt.json',{'policy_qids':[q['qid'] for q in training],
        'policy_groups':sorted({q['group'] for q in training}),'frozen_ranker_sha256':sha(ROOT/'ranker.pkl'),
        'all_policy_training_source_edges_hidden':True})
    path=ROOT/'policy_training_cases.pkl'
    cases=load(path) if path.exists() else generate(engine,contexts,training,variant,ranker,config['policy_conditions'],'Policy training')
    if not path.exists():dump(path,cases)
    models=fit_policies(cases,set(map(tuple,data['edges'])),config);dump(ROOT/'utility_models.pkl',models)
    queries=queries_for(data,'validation');qmap={q['qid']:q for q in queries}
    path=ROOT/'policy_validation_cases.pkl'
    validation=load(path) if path.exists() else generate(engine,contexts,queries,variant,ranker,config['validation_conditions'],'Policy validation')
    if not path.exists():dump(path,validation)
    specs=[]
    for name,model in models.items():
        thresholds=config['utility_thresholds'] if model['kind']=='utility' else config['wrongness_thresholds']
        specs.extend({'model':name,'kind':model['kind'],'budget':b,'threshold':t} for b in config['deletion_budgets'] for t in thresholds)
    gains=np.zeros(len(specs));actions_count=np.zeros(len(specs))
    for index,case in enumerate(validation):
        query=qmap[case['qid']];ctx=contexts[case['condition']]
        predictions={n:utility_predictions(m,case['features']) for n,m in models.items()}
        acts=[tuple(choose_actions(case,predictions[s['model']],s['budget'],s['threshold'])) for s in specs]
        batches=sorted({a for a in acts if len(a)>1})
        base=engine.retrieve(query,ctx,variant)
        changed=[engine.retrieve(query,ctx,variant,a) for a in batches]
        rankings=batch_rank(ranker,changed,base,case['base_scores'])
        baseline=metrics(case['base_ranking'],set(query['gold']),20)['recall']
        effects={a:metrics(r,set(query['gold']),20)['recall']-baseline for a,r in zip(batches,rankings)}
        effects[()]=0
        effects.update({(e,):float(v[1]) for e,v in zip(case['edges'],case['benefits'])})
        for j,a in enumerate(acts):gains[j]+=effects[a];actions_count[j]+=len(a)
        engine.trim_cache()
        if (index+1)%250==0:print(f'Policy selection {index+1}/{len(validation)}',flush=True)
    grid=[{**s,'validation_gain':float(gains[i]/len(validation)),
        'validation_mean_deletions':float(actions_count[i]/len(validation))} for i,s in enumerate(specs)]
    write_json(ROOT/'stage3_validation_grid.json',grid)
    policies={k:max([r for r in grid if r['kind']==k],
        key=lambda r:(round(r['validation_gain'],12),-r['validation_mean_deletions'],-r['budget'])) for k in ['utility','wrongness']}
    write_json(ROOT/'stage3_selection.json',{'policies':policies,'selected_before_test':True,
        'stage2_sha256':sha(ROOT/'stage2_selection.json')})
    print('Policy locked '+json.dumps(policies),flush=True)


def test(config,data,contexts,engine):
    if (ROOT/'run_complete.json').exists():raise RuntimeError('Preserve completed test')
    lock_path=ROOT/'test_lock.json'
    paths=[CONFIG,Path('RESTART_V3_PROTOCOL.md'),Path(config['dataset']),ROOT/'index_manifest.json']
    paths += [ROOT/f'stage{i}_selection.json' for i in [1,2,3]]
    paths += [ROOT/n for n in ['ranker.pkl','text_ranker.pkl','utility_models.pkl','graphs.pkl','vectorizer.pkl']]
    paths += [ROOT/'environment.json']
    paths += [Path('owe')/n for n in ['restart_v3.py','restart_support.py','restart_workers.py',
        'sequential_retriever.py','sequential_interventions.py','sequential_stage12.py','retrieval.py','experiment.py']]
    frozen={p.as_posix():sha(p) for p in paths}
    if lock_path.exists():assert json.loads(lock_path.read_text())['files']==frozen
    else:write_json(lock_path,{'files':frozen,'selected_before_test':True,'fresh_signature_test':True})
    variant=json.loads((ROOT/'stage1_selection.json').read_text())['variant']
    policies=json.loads((ROOT/'stage3_selection.json').read_text())['policies']
    ranker=load(ROOT/'ranker.pkl');text=load(ROOT/'text_ranker.pkl');models=load(ROOT/'utility_models.pkl')
    queries=queries_for(data,'test');reference=contexts['clean']['edges'];started=time.perf_counter()
    for condition in config['test_conditions']:
        output=ROOT/f'test_{condition}.csv'
        if output.exists():print('Retained '+condition,flush=True);continue
        case_path=ROOT/f'test_{condition}_cases.pkl'
        cases=load(case_path) if case_path.exists() else generate(engine,contexts,queries,variant,ranker,[condition],'Test '+condition)
        for case in cases:
            case['source_supported']=[e in reference for e in case['edges']]
            case['wrongness_scores']=utility_predictions(models['wrongness_boost'],case['features'])
        dump(case_path,cases)
        rows=[]
        with gzip.open(ROOT/f'test_{condition}_rankings.jsonl.gz','wt',encoding='utf-8') as f:
            for query,case in zip(queries,cases):
                assert query['qid']==case['qid']
                context=query_context(engine,query,contexts,condition,variant);base=engine.retrieve(query,context,variant)
                scores=case['base_scores'];info=engine.query_info(query);gold=set(query['gold'])
                clean=engine.retrieve(query,contexts['clean'],variant)
                cleanrank=predicted_ranking(clean,model_scores(ranker,clean.features))[:40]
                acts={k:choose_actions(case,utility_predictions(models[p['model']],case['features']),p['budget'],p['threshold']) for k,p in policies.items()}
                rng=np.random.default_rng(stable_int(query['qid']+condition+'v3-matched'))
                order=rng.permutation(len(case['edges']))
                for k in list(acts):acts['matched_'+k]=[case['edges'][int(i)] for i in order[:len(acts[k])]]
                if len(case['edges']):
                    best=int(np.argmax(case['benefits'][:,1]));acts['oracle1']=[case['edges'][best]] if case['benefits'][best,1]>0 else []
                else:acts['oracle1']=[]
                singles={e:r for e,r in zip(case['edges'],case['single_rankings'])}
                batches=sorted({tuple(a) for a in acts.values() if len(a)>1})
                changed=[engine.retrieve(query,context,variant,a) for a in batches]
                ranks=batch_rank(ranker,changed,base,scores);batchmap=dict(zip(batches,ranks))
                # Restricted legacy traversal prevents same-signature documents.
                eligible=info['eligible']
                class Restricted:
                    def __getitem__(self,u):return [v for v in context['adjacency'][u] if eligible[v]]
                old_neighborhood=Neighborhood(Restricted(),info['lexical'][:4],info['scores'])
                old=old_neighborhood.ranking[:40]
                small_variant={**variant,'lexical':32,'prior':16}
                original_cap=engine.config['candidate_cap'];engine.config['candidate_cap']=64
                try:small=engine.retrieve(query,context,small_variant)
                finally:engine.config['candidate_cap']=original_cap
                rankings={'lexical':info['lexical'][:40],'old_local':old,
                    'expanded_text':predicted_ranking(base,base.features[:,0])[:40],
                    'text_ranker':predicted_ranking(base,model_scores(text,base.features))[:40],
                    'proof_ranker':case['base_ranking'],
                    'proof_small64':predicted_ranking(small,model_scores(ranker,small.features))[:40],
                    'source_verified':cleanrank if condition.startswith(('random','hard','local')) else case['base_ranking']}
                verified_result=clean if condition.startswith(('random','hard','local')) else base
                if condition.startswith('mixed'):
                    verified=context_from_edges(engine.nodes,context['edges']&reference,'verified/'+condition+'/'+query['qid'])
                    vr=engine.retrieve(query,verified,variant)
                    verified_result=vr
                    rankings['source_verified']=predicted_ranking(vr,model_scores(ranker,vr.features))[:40]
                for k,a in acts.items():rankings[k]=case['base_ranking'] if not a else singles[a[0]] if len(a)==1 else batchmap[tuple(a)]
                gold_hits=len(set(base.ids)&gold)
                for method,ids in rankings.items():
                    assert len(set(map(int,ids)))==len(ids)
                    assert all(0<=int(v)<query['node'] and engine.nodes[int(v)]['signature']!=query['signature'] for v in ids)
                    selected=acts.get(method,[])
                    # Candidate counters for policy methods remain the base pool;
                    # direct baselines and the small recipe have their own pools.
                    pool_ids=base.ids;pool_transitions=base.transitions
                    if method=='lexical':pool_ids=info['lexical'][:config['candidate_cap']];pool_transitions=0
                    elif method=='old_local':pool_ids=old_neighborhood.ranking;pool_transitions=len(old_neighborhood.edges)
                    elif method=='proof_small64':pool_ids=small.ids;pool_transitions=small.transitions
                    elif method=='source_verified':pool_ids=verified_result.ids;pool_transitions=verified_result.transitions
                    pool_hits=len(set(map(int,pool_ids))&gold)
                    for budget in config['output_budgets']:
                        measured=metrics(ids,gold,budget,engine.lengths)
                        baseline=metrics(case['base_ranking'],gold,budget)['recall']
                        rows.append({'qid':query['qid'],'group':query['group'],'topic':query['topic'],
                            'condition':condition,'method':method,'budget':budget,**measured,
                            'delta_vs_none':measured['recall']-baseline,'deletions':len(selected),
                            'wrong_deletions':sum(e not in reference for e in selected),
                            'candidate_coverage':pool_hits/len(gold),'candidate_ceiling':min(budget,pool_hits)/len(gold),
                            'candidates':len(pool_ids),'transitions':pool_transitions,
                            'exposed_false_edges':case['exposed_false_edges'],'injected_edges':case['injected_edges'],
                            'offline_seconds':case['offline_seconds']})
                f.write(json.dumps({'qid':query['qid'],'condition':condition,'actions':acts,
                    'candidates':list(map(int,base.ids)),'rankings':{k:list(map(int,v)) for k,v in rankings.items()}})+'\n')
                engine.trim_cache()
        pd.DataFrame(rows).to_csv(output,index=False)
        print(f'Test complete {condition}: {len(rows)} rows',flush=True)
    combined=pd.concat([pd.read_csv(ROOT/f'test_{c}.csv') for c in config['test_conditions']],ignore_index=True)
    combined.to_csv(ROOT/'results.csv',index=False)
    assert all(sha(p)==s for p,s in frozen.items())
    write_json(ROOT/'run_complete.json',{'test_queries':len(queries),'conditions':len(config['test_conditions']),
        'methods':combined.method.nunique(),'measurement_rows':len(combined),'test_seconds':time.perf_counter()-started,
        'test_previously_indexed_signature_count':0,'all_selections_locked_before_test':True})
    print('All fresh test conditions completed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['index','candidates','rank','policy','test','infer'])
    parser.add_argument('--qid',default=None);parser.add_argument('--condition',default='clean')
    parser.add_argument('--policy',choices=['none','utility','wrongness'],default='none');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'environment.json').exists():
        write_json(ROOT/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
            'pandas':pd.__version__,'threadpoolctl':threadpoolctl.__version__,
            'logical_cpu_count':os.cpu_count(),'fit_threads':4,'intervention_workers':3,
            'inference_openmp_threads':1,'paid_model_api_used':False})
    with threadpool_limits(limits=4):
        if args.stage=='index':initialize();return
        config,data,contexts,engine=load_study()
        if args.stage=='candidates':candidates(config,data,contexts,engine)
        elif args.stage=='rank':rank(config,data,contexts,engine)
        elif args.stage=='policy':policy(config,data,contexts,engine)
        elif args.stage=='test':test(config,data,contexts,engine)
        else:
            q=next(q for q in data['queries'] if q['qid']==args.qid)
            variant=json.loads((ROOT/'stage1_selection.json').read_text())['variant']
            policy_spec=None if args.policy=='none' else json.loads((ROOT/'stage3_selection.json').read_text())['policies'][args.policy]
            if policy_spec:policy_spec={**policy_spec,'condition_name':args.condition}
            ctx=query_context(engine,q,contexts,args.condition,variant)
            result=infer(engine,q,ctx,variant,load(ROOT/'ranker.pkl'),policy_spec,load(ROOT/'utility_models.pkl'))
            print(json.dumps({'qid':q['qid'],'condition':args.condition,'actions':result['actions'],'seconds':result['seconds'],
                'retrieved':[engine.nodes[int(v)] for v in result['ranking'][:20]]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
