import numpy as np, random, json, chunk as C
from embed import embed_texts, embed_queries
from utils import iter_entries, PUBLIC_QUERIES_PATH
from eval import ndcg_at_k, load_query_file

random.seed(0)
rows=load_query_file(PUBLIC_QUERIES_PATH)
queries=[r["query"] for r in rows]; gt=[r["relevant_page_ids"] for r in rows]
gtpages=set().union(*gt)
recs=list(iter_entries())
byid={r['page_id']:r for r in recs}
gtpages={p for p in gtpages if p in byid}
others=[r for r in recs if r['page_id'] not in gtpages]
random.shuffle(others)
distract=others[:8000]
build=[byid[p] for p in gtpages]+distract
print("building on",len(build),"pages (",len(gtpages),"gt +",len(distract),"distract)",flush=True)
chunks=C.chunk_corpus(build, show_progress=True)
texts=[c.text for c in chunks]; pids=np.array([c.page_id for c in chunks])
print("embedding",len(texts),"chunks",flush=True)
vecs=embed_texts(texts, show_progress=True)
qv=embed_queries(queries)
np.save('sc_vecs.npy',vecs); np.save('sc_pids.npy',pids); np.save('sc_qv.npy',qv)
distract_order=[r['page_id'] for r in distract]

def eval_pool(npool):
    keep=set(gtpages)|set(distract_order[:npool])
    mask=np.array([p in keep for p in pids])
    v=vecs[mask]; pp=pids[mask]
    dense=qv@v.T
    ns=[]
    for qi in range(len(queries)):
        pb={}
        for ci,s in enumerate(dense[qi]):
            pid=int(pp[ci])
            if s>pb.get(pid,-1e9): pb[pid]=s
        order=sorted(pb,key=pb.get,reverse=True)[:10]
        ns.append(ndcg_at_k(order,gt[qi]))
    return len(keep), float(np.mean(ns))

print("=== SCALING CURVE (dense maxpool) ===",flush=True)
for npool in [0,250,500,1000,2000,4000,8000]:
    npages,nd=eval_pool(npool)
    print(f"pool_pages={npages:5d}  meanNDCG@10={nd:.4f}",flush=True)
