import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT=Path("/home/cunyuliu/pc_cng_research/results")
runs=[p.name for p in ROOT.glob("full_feasibility_mlp_*e*") if (p/"metrics.json").exists() and (p/"val_predictions.csv").exists() and "smoke" not in p.name]
runs=sorted(runs)

def read_pred(path):
    y=[]; s=[]; rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r); y.append(int(r["label"])); s.append(float(r["score"]))
    return rows, np.array(y), np.array(s)

def metrics_at(y,s,t):
    p=(s>=t).astype(int)
    out={"threshold":float(t),"accuracy":float(accuracy_score(y,p)),"f1":float(f1_score(y,p,zero_division=0)),"pred_positive_rate":float(np.mean(p)),"n":int(len(y))}
    if len(set(y.tolist()))>1:
        out["roc_auc"]=float(roc_auc_score(y,s)); out["auprc"]=float(average_precision_score(y,s))
    return out

def best_f1_threshold(y,s):
    thresholds=np.unique(np.quantile(s, np.linspace(0.01,0.99,199)))
    best=(0.5,-1.0)
    for t in thresholds:
        score=f1_score(y,(s>=t).astype(int),zero_division=0)
        if score>best[1]: best=(float(t),float(score))
    return best[0]

summary=[]
for run in runs:
    d=ROOT/run
    m=json.load(open(d/"metrics.json"))
    _, yv, sv=read_pred(d/"val_predictions.csv")
    _, yt, st=read_pred(d/"test_predictions.csv")
    t=best_f1_threshold(yv,sv)
    rec={
      "run": run,
      "train_rows_featurized": m["counts"]["train_rows_featurized"],
      "train_positive": m["counts"]["train_positive"],
      "train_negative": m["counts"]["train_negative"],
      "val_default": m["val"],
      "test_default": m["test"],
      "val_best_f1_threshold": metrics_at(yv,sv,t),
      "test_at_val_best_f1_threshold": metrics_at(yt,st,t),
      "test_by_dataset": m.get("test_by_dataset",{}),
      "test_by_reaction_class": m.get("test_by_reaction_class",{}),
      "best_checkpoint": m["best_checkpoint"],
    }
    summary.append(rec)

out={
  "selection": {
    "best_by_val_roc_auc": max(summary, key=lambda r: r["val_default"].get("roc_auc", -1))["run"],
    "best_by_val_auprc": max(summary, key=lambda r: r["val_default"].get("auprc", -1))["run"],
    "best_by_val_f1": max(summary, key=lambda r: r["val_best_f1_threshold"].get("f1", -1))["run"],
  },
  "runs": summary,
}
path=ROOT/"full_feasibility_matrix_summary.json"
json.dump(out, open(path,"w"), indent=2, ensure_ascii=False)
print(json.dumps(out["selection"], indent=2))
for r in sorted(summary, key=lambda x: x["val_default"].get("roc_auc", -1), reverse=True):
    print(r["run"], "val_auc", r["val_default"].get("roc_auc"), "test_auc", r["test_default"].get("roc_auc"), "test_auprc", r["test_default"].get("auprc"), "test_f1@0.5", r["test_default"].get("f1"), "test_f1@best", r["test_at_val_best_f1_threshold"].get("f1"))
print(path)
