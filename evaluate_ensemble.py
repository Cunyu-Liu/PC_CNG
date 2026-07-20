import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT=Path("/home/cunyuliu/pc_cng_research/results")
runs=[
 "full_feasibility_mlp_real_only_h2048_n4096_e80",
 "full_feasibility_mlp_real_only_h4096_n2048_e80",
 "full_feasibility_mlp_real_only_h2048_e60",
]

def read(path):
    rows=[]; y=[]; s=[]
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r); y.append(int(r["label"])); s.append(float(r["score"]))
    return rows, np.asarray(y), np.asarray(s)

def metrics(y,s,thr=0.5):
    p=(s>=thr).astype(int)
    out={"accuracy":float(accuracy_score(y,p)),"f1":float(f1_score(y,p,zero_division=0)),"pred_positive_rate":float(np.mean(p)),"n":int(len(y)),"roc_auc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s)),"threshold":float(thr)}
    return out

def best_thr(y,s):
    best=(0.5,-1)
    for t in np.unique(np.quantile(s, np.linspace(0.01,0.99,199))):
        f=f1_score(y,(s>=t).astype(int),zero_division=0)
        if f>best[1]: best=(float(t),float(f))
    return best[0]

val_rows,yv,sv0=read(ROOT/runs[0]/"val_predictions.csv")
test_rows,yt,st0=read(ROOT/runs[0]/"test_predictions.csv")
val_scores=[]; test_scores=[]
for run in runs:
    _, y, s=read(ROOT/run/"val_predictions.csv")
    assert np.array_equal(y,yv), run
    val_scores.append(s)
    _, y, s=read(ROOT/run/"test_predictions.csv")
    assert np.array_equal(y,yt), run
    test_scores.append(s)
val_scores=np.stack(val_scores)
test_scores=np.stack(test_scores)

candidates=[]
# single models and pair/simplex grid, optimized on val ROC-AUC then AUPRC.
grids=[]
for i in range(len(runs)):
    w=np.zeros(len(runs)); w[i]=1; grids.append(w)
for a in np.linspace(0,1,101):
    w=np.array([a,1-a,0.0]); grids.append(w)
    w=np.array([a,0.0,1-a]); grids.append(w)
    w=np.array([0.0,a,1-a]); grids.append(w)
for a in np.linspace(0,1,51):
  for b in np.linspace(0,1-a,51):
    c=1-a-b
    if c < -1e-9: continue
    grids.append(np.array([a,b,c]))

for w in grids:
    sv=(w[:,None]*val_scores).sum(axis=0)
    st=(w[:,None]*test_scores).sum(axis=0)
    t=best_thr(yv,sv)
    candidates.append({"weights":w.tolist(),"runs":runs,"val":metrics(yv,sv),"val_best_f1":metrics(yv,sv,t),"test":metrics(yt,st),"test_at_val_best_f1":metrics(yt,st,t)})

best_auc=max(candidates, key=lambda r:(r["val"]["roc_auc"], r["val"].get("auprc",0)))
best_auprc=max(candidates, key=lambda r:(r["val"]["auprc"], r["val"].get("roc_auc",0)))
best_f1=max(candidates, key=lambda r:r["val_best_f1"]["f1"])
out={"best_by_val_auc":best_auc,"best_by_val_auprc":best_auprc,"best_by_val_f1":best_f1}
out_path=ROOT/"ensemble_real_only_summary.json"
json.dump(out, open(out_path,"w"), indent=2)
for name, rec in out.items():
    print(name, rec["weights"])
    print(" val", rec["val"])
    print(" test", rec["test"])
    print(" test@valF1", rec["test_at_val_best_f1"])
print(out_path)
