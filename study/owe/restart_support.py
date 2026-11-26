"""Small importable helpers for the independently fitted restart study."""
from __future__ import annotations
import numpy as np
from .prepare import stable_int
from .sequential_retriever import prepare_contexts

GRAPH_COLUMNS = [5,6,20,21,22,23,24,25,26,27]
TEXT_COLUMNS = [i for i in range(42) if i not in GRAPH_COLUMNS]


class ProjectedEstimator:
    def __init__(self, estimator, columns):
        self.estimator = estimator
        self.columns = np.asarray(columns, dtype=int)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x[:,self.columns])


def context_from_edges(nodes, edges, name, changes=()):
    edges = set(map(tuple,edges))
    adjacency=[[] for _ in nodes];incoming=[[] for _ in nodes]
    for u,v in sorted(edges):
        assert u>v
        adjacency[u].append(v);incoming[v].append(u)
    c={'name':name,'edges':edges,'adjacency':adjacency,'incoming':incoming,
       'audit_changes':list(changes)}
    return prepare_contexts({name:c})[name]


def local_context(engine,query,clean,variant,count=8):
    """Gold-blind stress; no effect-driven question or edge selection."""
    public_query={k:v for k,v in query.items() if not k.startswith('gold')}
    result=engine.retrieve(public_query,clean,variant)
    scores=engine.query_info(public_query)['scores']
    sources=sorted({u for u,v in result.edges},key=lambda u:(-scores[u],u))
    additions=[]
    for u in sources:
        targets=[int(v) for v in engine.query_info(public_query)['lexical']
                 if v<u and engine.nodes[int(v)]['core']==engine.nodes[u]['core']
                 and (u,int(v)) not in clean['edges']]
        if not targets:continue
        # Near-source and query-relevant choices are observable, not gold.
        targets=targets[:128]
        near=(engine.matrix[targets]@engine.matrix[u].T).toarray().ravel()
        order=np.lexsort((np.asarray(targets),-near))
        additions.append((u,targets[int(order[0])]))
        if len(additions)>=count:break
    return context_from_edges(engine.nodes,clean['edges']|set(additions),
        'local8_s439/'+query['qid'],[('added',u,v) for u,v in additions])


def query_context(engine,query,contexts,condition,variant):
    if condition=='local8_s439':
        return local_context(engine,query,contexts['clean'],variant,engine.config['local_injection_count'])
    return contexts[condition]
