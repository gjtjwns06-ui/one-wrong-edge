"""Run the frozen retriever on an existing formal query without loading its gold into inference."""
from __future__ import annotations
import argparse
import json
import pickle
import time

from .sequential_stage12 import ROOT,load_study
from .sequential_retriever import model_scores,predicted_ranking
from .sequential_interventions import candidate_edges,edge_features,utility_predictions,choose_actions


def infer(engine,query,context,condition,variant,ranker,policy=None,policy_model=None):
    query={key:value for key,value in query.items() if key!='gold'}
    start=time.perf_counter()
    base=engine.retrieve(query,context,variant)
    scores=model_scores(ranker,base.features)
    actions=[]
    if policy is not None and policy['threshold']<1:
        edges=list(base.edges) if policy.get('scope')=='all_traversed' else candidate_edges(base,query,condition,scores,engine.config['utility_candidate_cap'])
        features=edge_features(engine,query,context,base,scores,edges)
        predictions=utility_predictions(policy_model,features)
        actions=choose_actions({'edges':edges},predictions,policy['budget'],policy['threshold'])
    if actions:
        result=engine.retrieve(query,context,variant,actions)
        scores=model_scores(ranker,result.features)
    else:
        result=base
    ranking=predicted_ranking(result,scores)
    return {'ranking':ranking,'actions':actions,'candidates':result.ids,'seconds':time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qid',required=True)
    parser.add_argument('--condition',default='clean')
    parser.add_argument('--policy',choices=['none','utility1','utility3','wrongness1','wrongness3','wrongness_filter'],default='none')
    parser.add_argument('--budget',type=int,default=20)
    args=parser.parse_args()
    config,data,contexts,engine=load_study()
    query=next(q for q in data['queries'] if q['qid']==args.qid)
    variant=json.loads((ROOT/'stage1_selection.json').read_text())['variant']
    ranker=pickle.loads((ROOT/'ranker.pkl').read_bytes())
    policy=None;policy_model=None
    if args.policy!='none':
        policy=json.loads((ROOT/'stage3_selection.json').read_text())['policies'][args.policy]
        policy_model=pickle.loads((ROOT/'utility_models.pkl').read_bytes())[policy['model']]
    result=infer(engine,query,contexts[args.condition],args.condition,variant,ranker,policy,policy_model)
    print(json.dumps({'qid':args.qid,'condition':args.condition,'policy':args.policy,'seconds':result['seconds'],
        'deleted_relations':[[data['nodes'][u]['label'],data['nodes'][v]['label']] for u,v in result['actions']],
        'retrieved':[{'label':data['nodes'][int(u)]['label'],'statement_and_hypotheses':data['nodes'][int(u)]['text'],
                      'source':data['nodes'][int(u)]['source']} for u in result['ranking'][:args.budget]]},indent=2))


if __name__=='__main__':main()
