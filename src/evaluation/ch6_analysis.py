ch6_code = '''#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

CONTROL    = "lstm_price"
EXPERIMENT = "fusion"

def regression_metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    err = y - yhat
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def pct_improvement(ctrl, model):
    return 100.0 * (ctrl - model) / ctrl

def diebold_mariano(e1, e2, loss="absolute", h=1, hln=True):
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d  = np.abs(e1) - np.abs(e2) if loss == "absolute" else e1**2 - e2**2
    n  = len(d); dbar = d.mean()
    lrv = np.mean((d - dbar)**2)
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2.0 * (1 - k/h) * gk
    dm = dbar / np.sqrt(lrv / n)
    if hln:
        dm *= np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        return float(dm), float(2*stats.t.cdf(-abs(dm), df=n-1)), float(stats.t.sf(dm, df=n-1))
    return float(dm), float(2*stats.norm.cdf(-abs(dm))), float(stats.norm.sf(dm))

def mcnemar_test(y_true, pred_a, pred_b):
    y = np.asarray(y_true)
    a_ok = (np.asarray(pred_a) == y); b_ok = (np.asarray(pred_b) == y)
    n01 = int(np.sum(a_ok & ~b_ok)); n10 = int(np.sum(~a_ok & b_ok))
    res = mcnemar([[0, n01],[n10, 0]], exact=False, correction=True)
    return float(res.statistic), float(res.pvalue), n01, n10

def save_table(df, path):
    try:
        df.to_markdown(path, index=False, floatfmt=".4f")
    except Exception:
        df.to_csv(path.replace(".md",".csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir",  default="data/processed")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--maxlag", type=int, default=4)
    ap.add_argument("--dm_loss", default="absolute", choices=["absolute","squared"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # RQ1 classification
    clf = pd.read_csv(os.path.join(args.indir, "classification.csv"))
    rows = []
    for m in [c for c in clf.columns if c != "y_true"]:
        rows.append({"Model": m,
                     "Accuracy":          accuracy_score(clf.y_true, clf[m]),
                     "Precision (macro)": precision_score(clf.y_true, clf[m], average="macro", zero_division=0),
                     "Recall (macro)":    recall_score(clf.y_true, clf[m], average="macro", zero_division=0),
                     "macro-F1":          f1_score(clf.y_true, clf[m], average="macro", zero_division=0)})
    clf_tbl = pd.DataFrame(rows)
    save_table(clf_tbl, os.path.join(args.outdir, "table_6_3_classification.md"))

    # RQ3 forecasting
    fc = pd.read_csv(os.path.join(args.indir, "forecasts.csv"))
    mae_ctrl, _ = regression_metrics(fc.y_true, fc[CONTROL])
    rows = []
    for m in [c for c in fc.columns if c not in ("date","y_true")]:
        mae, rmse = regression_metrics(fc.y_true, fc[m])
        rows.append({"Model": m, "MAE": mae, "RMSE": rmse,
                     "MAE %impr vs control": pct_improvement(mae_ctrl, mae)})
    fc_tbl = pd.DataFrame(rows)
    save_table(fc_tbl, os.path.join(args.outdir, "table_6_4_forecasting.md"))

    # DM test
    e_ctrl = (fc.y_true - fc[CONTROL]).values
    e_exp  = (fc.y_true - fc[EXPERIMENT]).values
    dm, p2, p1 = diebold_mariano(e_ctrl, e_exp, loss=args.dm_loss)

    # RQ2 Granger
    s = pd.read_csv(os.path.join(args.indir, "series_daily.csv")).dropna()
    gres = grangercausalitytests(s[["price","sentiment"]].values, maxlag=args.maxlag, verbose=False)
    g_rows = [{"lag": lag, "F": gres[lag][0]["ssr_ftest"][0],
               "p-value": gres[lag][0]["ssr_ftest"][1]} for lag in range(1, args.maxlag+1)]
    granger_tbl = pd.DataFrame(g_rows)
    save_table(granger_tbl, os.path.join(args.outdir, "table_6_granger.md"))

    # McNemar
    mcn = []
    if "laft_afriberta" in clf.columns:
        for base in [c for c in ["b2_transformer_nolaft","b1_svm_tfidf"] if c in clf.columns]:
            stat, p, n01, n10 = mcnemar_test(clf.y_true, clf[base], clf["laft_afriberta"])
            mcn.append({"comparison": f"LAFT vs {base}", "chi2": stat,
                        "p-value": p, "n(base only)": n01, "n(LAFT only)": n10})

    # Significance summary
    sig = [
        {"Research question": "RQ3 H0: MAE_multi >= MAE_control",
         "Test": f"Diebold-Mariano ({args.dm_loss})", "Statistic": dm, "p-value": p1,
         "Decision": "reject H0" if p1 < 0.05 else "fail to reject H0"},
        {"Research question": "RQ2 sentiment does NOT Granger-cause price",
         "Test": f"Granger F (min p, lags 1..{args.maxlag})",
         "Statistic": float(granger_tbl.F.max()),
         "p-value": float(granger_tbl["p-value"].min()),
         "Decision": "reject H0" if granger_tbl["p-value"].min() < 0.05 else "fail to reject H0"},
    ]
    if mcn:
        best = pd.DataFrame(mcn).iloc[pd.DataFrame(mcn)["p-value"].idxmin()]
        sig.append({"Research question": "RQ1 LAFT = best baseline",
                    "Test": "McNemar", "Statistic": float(best["chi2"]),
                    "p-value": float(best["p-value"]),
                    "Decision": "reject H0" if best["p-value"] < 0.05 else "fail to reject H0"})
    sig_tbl = pd.DataFrame(sig)
    save_table(sig_tbl, os.path.join(args.outdir, "table_6_5_significance.md"))

    print("=== RQ1 Classification ===")
    print(clf_tbl.to_string(index=False))
    print("\\n=== RQ3 Forecasting ===")
    print(fc_tbl.to_string(index=False))
    print(f"\\nDiebold-Mariano: DM={dm:.4f}  p(fusion better)={p1:.4f}")
    print("\\n=== RQ2 Granger ===")
    print(granger_tbl.to_string(index=False))
    if mcn:
        print("\\n=== McNemar ===")
        print(pd.DataFrame(mcn).to_string(index=False))
    print("\\n=== Significance summary ===")
    print(sig_tbl.to_string(index=False))
    print(f"\\n✓ All tables written to: {args.outdir}/")

if __name__ == "__main__":
    main()
'''

with open("src/evaluation/ch6_analysis.py", "w") as f:
    f.write(ch6_code)
print("✓ ch6_analysis.py written")
