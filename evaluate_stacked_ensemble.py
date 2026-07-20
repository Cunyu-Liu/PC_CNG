import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=Path("/home/cunyuliu/pc_cng_research/results")
runs=[]
for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
        continue
    if (d/"val_predictions.csv").exists() and (d/"test_predictions.csv").exists() and "smoke" not in d.name:
        runs.append(d.name)

def read(path):
    rows=[]; y=[]; s=[]
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r); y.append(int(r["label"])); s.append(float(r["score"]))
    return rows, np.asarray(y), np.asarray(s)

def metrics(y,s,t=0.5):
    p=(s>=t).astype(int)
    return {"accuracy":float(accuracy_score(y,p)),"f1":float(f1_score(y,p,zero_division=0)),"pred_positive_rate":float(np.mean(p)),"n":int(len(y)),"roc_auc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s)),"threshold":float(t)}

def best_thr(y,s):
    best=(0.5,-1)
    for t in np.unique(np.quantile(s, np.linspace(0.01,0.99,199))):
        f=f1_score(y,(s>=t).astype(int),zero_division=0)
        if f>best[1]: best=(float(t),float(f))
    return best[0]

val_rows,yv,first=read(ROOT/runs[0]/"val_predictions.csv")
test_rows,yt,firstt=read(ROOT/runs[0]/"test_predictions.csv")
Xv=[]; Xt=[]; used=[]
for run in runs:
    _, y, s=read(ROOT/run/"val_predictions.csv")
    _, yt2, st=read(ROOT/run/"test_predictions.csv")
    if len(y)!=len(yv) or len(yt2)!=len(yt) or not np.array_equal(y,yv) or not np.array_equal(yt2,yt):
        continue
    Xv.append(s); Xt.append(st); used.append(run)
Xv=np.stack(Xv, axis=1)
Xt=np.stack(Xt, axis=1)

records=[]
# Conservative convex average of top val-AUC models.
val_aucs=[roc_auc_score(yv,Xv[:,i]) for i in range(Xv.shape[1])]
top_idx=np.argsort(val_aucs)[-5:]
for n in [2,3,5]:
    idx=top_idx[-n:]
    sv=Xv[:,idx].mean(axis=1); st=Xt[:,idx].mean(axis=1); thr=best_thr(yv,sv)
    records.append({"method":f"mean_top{n}","runs":[used[i] for i in idx],"val":metrics(yv,sv),"test":metrics(yt,st),"test_at_val_f1":metrics(yt,st,thr)})

for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0]:
    clf=make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10000, class_weight="balanced", solver="lbfgs"))
    clf.fit(Xv,yv)
    sv=clf.predict_proba(Xv)[:,1]
    st=clf.predict_proba(Xt)[:,1]
    thr=best_thr(yv,sv)
    records.append({"method":f"logreg_balanced_C{C}","runs":used,"val":metrics(yv,sv),"test":metrics(yt,st),"test_at_val_f1":metrics(yt,st,thr)})

for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0]:
    clf=make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10000, solver="lbfgs"))
    clf.fit(Xv,yv)
    sv=clf.predict_proba(Xv)[:,1]
    st=clf.predict_proba(Xt)[:,1]
    thr=best_thr(yv,sv)
    records.append({"method":f"logreg_C{C}","runs":used,"val":metrics(yv,sv),"test":metrics(yt,st),"test_at_val_f1":metrics(yt,st,thr)})

best_auc=max(records, key=lambda r:(r["val"]["roc_auc"], r["val"]["auprc"]))
best_auprc=max(records, key=lambda r:(r["val"]["auprc"], r["val"]["roc_auc"]))
best_f1=max(records, key=lambda r:r["val"].get("f1",-1))
out={"used_runs":used,"best_by_val_auc":best_auc,"best_by_val_auprc":best_auprc,"best_by_val_f1_at_05":best_f1,"all_records":records}
out_path=ROOT/"stacked_ensemble_summary.json"
json.dump(out, open(out_path,"w"), indent=2)
for k in ["best_by_val_auc","best_by_val_auprc","best_by_val_f1_at_05"]:
    r=out[k]
    print(k, r["method"])
    print(" val", r["val"])
    print(" test", r["test"])
    print(" test@valF1", r["test_at_val_f1"])
print("used", len(used), used)
print(out_path)
