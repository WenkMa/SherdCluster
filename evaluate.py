#!/usr/bin/env python3
"""Evaluate descriptors and/or a finished partition. Labels never enter fitting."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch


def read_pairs(path):
    with Path(path).open(newline='') as f:
        pairs=[(r['id_a'],r['id_b']) for r in csv.DictReader(f)]
    if any(a==b for a,b in pairs) or len({tuple(sorted(p)) for p in pairs})!=len(pairs):
        raise ValueError('Reference pairs must be distinct, undirected and non-self')
    return pairs


def load_features(path):
    with np.load(path,allow_pickle=False) as z:
        ids=z['sample_ids'].astype(str);x=z['features'].astype(np.float32)
    if x.ndim!=2 or len(x)!=len(ids) or len(set(ids))!=len(ids):raise ValueError('Invalid feature bank')
    order=np.argsort(ids,kind='stable'); ids=ids[order];x=x[order]
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (norms<=0).any():raise ValueError('Invalid feature values')
    return ids,x/norms


def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate_retrieval(feature_path,pairs,device,cutoffs=(1,5,10),batch_size=64):
    ids,x=load_features(feature_path);ix={s:i for i,s in enumerate(ids)}
    targets=defaultdict(set)
    for a,b in pairs:targets[a].add(b);targets[b].add(a)
    absent=set(targets)-set(ids)
    if absent:raise ValueError(f'Reference sherds absent from feature bank: {sorted(absent)}')
    queries=sorted(targets);xt=torch.from_numpy(x).to(device);details=[];ranks=[]
    with torch.inference_mode():
        for start in range(0,len(queries),batch_size):
            qs=queries[start:start+batch_size]
            mat=(xt[[ix[q] for q in qs]]@xt.T).cpu().numpy()
            for q,scores in zip(qs,mat):
                scores[ix[q]]=-np.inf
                order=np.argsort(-scores,kind='stable')
                rr=np.empty(len(ids),np.int32);rr[order]=np.arange(1,len(ids)+1)
                best=min(int(rr[ix[t]]) for t in targets[q]);ranks.append(best)
                details.append(dict(query_id=q,partners=';'.join(sorted(targets[q])),best_partner_rank=best,
                                    **{f'top_{j+1}':str(ids[n]) for j,n in enumerate(order[:10])}))
    result=dict(feature_file=str(feature_path),gallery_size=len(ids),query_count=len(queries),reference_pairs=len(pairs),
                dimension=x.shape[1],self_excluded=True,tie_break='descending cosine, lexical sherd ID',
                hit_counts={str(k):int((np.array(ranks)<=k).sum()) for k in cutoffs},
                hit_percent={str(k):float(100*(np.array(ranks)<=k).mean()) for k in cutoffs})
    return result,details


def evaluate_partition(path,pairs):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    c={r['sample_id']:r['cluster_id'] for r in rows}
    if len(c)!=len(rows):raise ValueError('Duplicate sherd assignments')
    missing={s for p in pairs for s in p}-set(c)
    if missing:raise ValueError(f'Reference IDs missing from partition: {sorted(missing)}')
    groups=defaultdict(list)
    for s,g in c.items():groups[g].append(s)
    sizes=[len(g) for g in groups.values()]
    if any(int(r['cluster_size'])!=len(groups[r['cluster_id']]) for r in rows):raise ValueError('Invalid cluster sizes')
    details=[dict(id_a=a,id_b=b,cluster_a=c[a],cluster_b=c[b],retained=c[a]==c[b]) for a,b in pairs]
    retained=sum(r['retained'] for r in details)
    return dict(total_sherds=len(c),clusters=len(groups),singletons=sizes.count(1),non_singleton_clusters=sum(n>1 for n in sizes),
                max_cluster_size=max(sizes),intra_cluster_pairs=sum(n*(n-1)//2 for n in sizes),
                reference_pairs=len(pairs),retained_pairs=retained,retention_percent=100*retained/len(pairs)),details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,nargs='*');p.add_argument('--assignments',type=Path)
    p.add_argument('--pairs',type=Path,default=Path('data/reference_pairs.csv'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if not a.features and not a.assignments:p.error('Provide features and/or assignments')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pairs=read_pairs(a.pairs);a.output.mkdir(parents=True,exist_ok=False);result={}
    if a.features:
        result['representation']=[]
        for path in a.features:
            r,d=evaluate_retrieval(path,pairs,a.device);result['representation'].append(r)
            write_csv(a.output/(path.stem+'_queries.csv'),d);print(json.dumps(r),flush=True)
    if a.assignments:
        r,d=evaluate_partition(a.assignments,pairs);result['clustering']=r
        write_csv(a.output/'reference_pair_outcomes.csv',d);print(json.dumps(r),flush=True)
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
