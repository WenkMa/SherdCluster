#!/usr/bin/env python3
"""Label-free reciprocal-neighbor / all-pairs agglomeration, without a size cap."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from evaluate import load_features,write_csv


def compute_scores(x,path,device,block_size=512):
    scores=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(x),len(x)))
    xt=torch.from_numpy(x).to(device)
    with torch.inference_mode():
        for i in range(0,len(x),block_size):
            scores[i:i+block_size]=(xt[i:i+block_size]@xt.T).cpu().numpy()
    scores.flush();return scores


def graph_edges(scores,k,eta,mutual=True):
    n=len(scores);k=min(k,n-1)
    if k<1:return [],np.empty((n,0),dtype=np.int32)
    neighbors=np.empty((n,k),dtype=np.int32)
    for i in range(n):
        row=np.array(scores[i]);row[i]=-np.inf
        neighbors[i]=np.argsort(-row,kind='stable')[:k]
    sets=[set(map(int,row)) for row in neighbors]
    edges=set()
    for i,row in enumerate(neighbors):
        for j in row:
            j=int(j)
            if not mutual or i in sets[j]:edges.add((min(i,j),max(i,j)))
    edges=[(i,j,float(scores[i,j])) for i,j in edges if scores[i,j]>=eta]
    edges.sort(key=lambda v:(-v[2],v[0],v[1]))
    return edges,neighbors


def merge_clusters(scores,edges,rho):
    parent=list(range(len(scores)));members={i:[i] for i in parent};decisions=Counter()
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j,w in edges:
        a,b=find(i),find(j)
        if a==b:decisions['already_same']+=1;continue
        A,B=members[a],members[b]
        if rho is not None and float(scores[np.ix_(A,B)].min())<rho:
            decisions['rejected_all_pairs']+=1;continue
        if len(A)<len(B):a,b=b,a
        parent[b]=a;members[a].extend(members.pop(b));decisions['accepted']+=1
    return sorted((sorted(g) for g in members.values()),key=lambda g:g[0]),dict(decisions)


def cluster(feature_path,output,device='cpu',k=10,eta=.5,rho=.55,mutual=True,collection_ids=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();ids,x=load_features(feature_path)
    extras=[]
    if collection_ids:
        all_ids=Path(collection_ids).read_text().splitlines()
        if len(set(all_ids))!=len(all_ids) or not set(ids)<=set(all_ids):raise ValueError('Invalid collection manifest')
        extras=sorted(set(all_ids)-set(ids))
    scores=compute_scores(x,output/'scores.npy',device)
    edges,neighbors=graph_edges(scores,k,eta,mutual);groups,decisions=merge_clusters(scores,edges,rho)
    rows=[];stats=[]
    for c,g in enumerate(groups):
        cid=f'C{c:06d}';minimum=None
        if len(g)>1:
            minimum=float('inf')
            # Bounded-memory exact invariant check, also supports unconstrained giant clusters.
            for start in range(0,len(g),128):
                block=scores[np.ix_(g[start:start+128],g)]
                for offset,row in enumerate(block):
                    tail=row[start+offset+1:]
                    if len(tail):minimum=min(minimum,float(tail.min()))
            if rho is not None and minimum<rho:raise AssertionError('All-pairs invariant violated')
        stats.append(dict(cluster_id=cid,size=len(g),intra_cluster_pairs=len(g)*(len(g)-1)//2,min_similarity=minimum))
        rows.extend(dict(sample_id=str(ids[i]),cluster_id=cid,cluster_size=len(g),status='feature_available') for i in g)
    for c,s in enumerate(extras,len(groups)):
        rows.append(dict(sample_id=s,cluster_id=f'C{c:06d}',cluster_size=1,status='no_dual_surface_descriptor'))
        stats.append(dict(cluster_id=f'C{c:06d}',size=1,intra_cluster_pairs=0,min_similarity=None))
    write_csv(output/'assignments.csv',rows);write_csv(output/'clusters.csv',stats)
    np.savez_compressed(output/'graph.npz',sample_ids=ids,neighbors=neighbors,
                        edges=np.array([(i,j) for i,j,w in edges],np.int32),weights=np.array([w for i,j,w in edges],np.float32))
    sizes=[s['size'] for s in stats];mins=[s['min_similarity'] for s in stats if s['min_similarity'] is not None]
    result=dict(feature_file=str(feature_path),feature_sherds=len(ids),total_sherds=len(rows),added_singletons=len(extras),
                k=k,eta=eta,rho=rho,mutual=mutual,max_cluster_size_limit=None,labels_read=False,
                neighbor_rule='mutual kNN' if mutual else 'symmetrized union kNN (either direction)',
                eligible_edges=len(edges),clusters=len(stats),singletons=sizes.count(1),max_cluster_size=max(sizes),
                intra_cluster_pairs=sum(s['intra_cluster_pairs'] for s in stats),minimum_similarity=min(mins) if mins else None,
                merge_decisions=decisions,seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');del scores
    (output/'scores.npy').unlink() # Large matrix is transient; no dependency on a precomputed graph.
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=Path('data/features/both_concat.npz'))
    p.add_argument('--collection-ids',type=Path,help='Full collection manifest; absent descriptors remain singletons')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--k',type=int,default=10);p.add_argument('--eta',type=float,default=.50)
    p.add_argument('--rho',type=float,default=.55);p.add_argument('--union-knn',action='store_true')
    p.add_argument('--no-all-pairs',action='store_true');a=p.parse_args()
    if a.k<1 or not -1<=a.eta<=1 or not -1<=a.rho<=1:p.error('Invalid clustering parameters')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=cluster(a.features,a.output,a.device,a.k,a.eta,None if a.no_all_pairs else a.rho,not a.union_knn,a.collection_ids)
    print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
