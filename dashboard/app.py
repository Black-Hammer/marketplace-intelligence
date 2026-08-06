"""
Marketplace Intelligence — interactive dashboard
================================================
Cross-lingual sentiment and price dynamics for Nigerian e-commerce.
MSc Data Science & Analytics · American University of Nigeria.

Run:   streamlit run app.py
Data:  point the sidebar at the folder holding classification.csv,
       series_daily.csv and forecasts.csv (usually data/processed), or upload
       them directly. Anything missing is filled with clearly-labelled
       demonstration data so the interface stays explorable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
import core

st.set_page_config(page_title="Marketplace Intelligence",
                   page_icon="◧", layout="wide",
                   initial_sidebar_state="expanded")
ch.install_theme()

# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------
# NOTE: no blank lines inside this block -- Markdown ends an HTML block at the
# first blank line, which would dump the remaining CSS onto the page as text.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#12211F; --muted:#5F726D; --line:#D7DEDB; --paper:#F3F5F4;
    --panel:#FFFFFF; --teal:#0E6B60; --marigold:#C98A0E; --plum:#6B3E7A;
    --clay:#A4271C; --slate:#7C8D95;
  }
  .stApp{background:var(--paper); color:var(--ink);
         font-family:'IBM Plex Sans',system-ui,sans-serif;}
  .block-container{padding-top:2.2rem; padding-bottom:4rem; max-width:1320px;}
  h1,h2,h3,h4{font-family:'Bricolage Grotesque',Georgia,serif; color:var(--ink);
              letter-spacing:-0.02em;}
  h1{font-size:2.35rem; line-height:1.05; margin:0;}
  h2{font-size:1.35rem; margin:0 0 .2rem 0;}
  h3{font-size:1.05rem;}
  .eyebrow{font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.16em;
           text-transform:uppercase; color:var(--muted);}
  .lede{color:var(--muted); font-size:.95rem; max-width:62ch; margin-top:.5rem;}
  .rule{height:1px; background:var(--line); margin:1.4rem 0 1.2rem;}
  /* provenance chip */
  .chip{display:inline-flex; gap:.5rem; align-items:center; font-family:'IBM Plex Mono',monospace;
        font-size:.72rem; padding:.32rem .7rem; border-radius:2px; border:1px solid var(--line);
        background:var(--panel); color:var(--muted);}
  .chip b{color:var(--ink); font-weight:500;}
  .chip.demo{border-color:#E7C77A; background:#FCF6E7; color:#8A6412;}
  /* stat cards */
  .cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px;}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:3px;
        padding:16px 18px 14px; position:relative; overflow:hidden;}
  .card:before{content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
               background:var(--accent,var(--teal));}
  .card .k{font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.13em;
           text-transform:uppercase; color:var(--muted);}
  .card .v{font-family:'Bricolage Grotesque',serif; font-size:1.9rem; font-weight:700;
           line-height:1.15; margin:.35rem 0 .1rem; font-variant-numeric:tabular-nums;}
  .card .n{font-size:.78rem; color:var(--muted); line-height:1.35;}
  /* early-warning banner */
  .warn{border:1px solid var(--line); border-radius:3px; background:var(--panel);
        padding:18px 22px; display:flex; align-items:center; gap:20px;}
  .warn .lvl{font-family:'Bricolage Grotesque',serif; font-size:1.6rem; font-weight:700;
             padding:.35rem 1.1rem; border-radius:2px; color:#fff;}
  .lvl.high{background:var(--clay);} .lvl.watch{background:var(--marigold);}
  .lvl.steady{background:var(--teal);}
  .warn .txt{font-size:.9rem; color:var(--muted);}
  .note{font-size:.8rem; color:var(--muted); border-left:2px solid var(--line);
        padding-left:.8rem; margin:.6rem 0 0;}
  /* tabs */
  .stTabs [data-baseweb="tab-list"]{gap:2px; border-bottom:1px solid var(--line);}
  .stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace; font-size:.74rem;
      letter-spacing:.08em; text-transform:uppercase; padding:10px 16px; color:var(--muted);}
  .stTabs [aria-selected="true"]{color:var(--ink); border-bottom:2px solid var(--teal);}
  section[data-testid="stSidebar"]{background:var(--panel); border-right:1px solid var(--line);}
  section[data-testid="stSidebar"] h2{font-size:1rem;}
  [data-testid="stMetricValue"]{font-family:'Bricolage Grotesque',serif;}
  .stButton>button{border-radius:2px; border:1px solid var(--ink); background:var(--ink);
                   color:#fff; font-weight:500; letter-spacing:.02em;}
  .stButton>button:hover{background:var(--teal); border-color:var(--teal); color:#fff;}
  :focus-visible{outline:2px solid var(--marigold) !important; outline-offset:2px;}
  @media (prefers-reduced-motion: reduce){*{animation:none !important; transition:none !important;}}
</style>
""", unsafe_allow_html=True)


# Streamlit renamed the container-width argument in 1.49; support both.
try:
    import inspect as _inspect
    _WIDE = ({"width": "stretch"}
             if "width" in _inspect.signature(st.dataframe).parameters
             else {"use_container_width": True})
except Exception:                                    # pragma: no cover
    _WIDE = {"use_container_width": True}


def card(label: str, value: str, note: str = "", accent: str = "var(--teal)") -> str:
    return (f"<div class='card' style='--accent:{accent}'><div class='k'>{label}</div>"
            f"<div class='v'>{value}</div><div class='n'>{note}</div></div>")


def cards(items: list[str]) -> None:
    st.markdown(f"<div class='cards'>{''.join(items)}</div>", unsafe_allow_html=True)


def section(title: str, blurb: str = "") -> None:
    st.markdown(f"<div class='rule'></div><h2>{title}</h2>"
                + (f"<div class='lede'>{blurb}</div>" if blurb else ""), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar — where the numbers come from
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='eyebrow'>Data source</div>", unsafe_allow_html=True)
    mode = st.radio("How to load results", ["Read a folder", "Upload files", "Demonstration only"],
                    label_visibility="collapsed")

    uploads: dict = {}
    root = "data/processed"
    if mode == "Read a folder":
        root = st.text_input("Folder holding the pipeline output", "data/processed",
                             help="Looks for classification.csv, series_daily.csv and forecasts.csv "
                                  "here, and in data/processed underneath it.")
    elif mode == "Upload files":
        uploads["classification"] = st.file_uploader("classification.csv", type="csv")
        uploads["series"] = st.file_uploader("series_daily.csv", type="csv")
        uploads["forecasts"] = st.file_uploader("forecasts.csv", type="csv")

    st.markdown("<div class='rule'></div><div class='eyebrow'>Reading window</div>",
                unsafe_allow_html=True)
    window = st.slider("Days of history to plot", 60, 420, 180, step=30)
    max_lag = st.slider("Maximum lag to test (days)", 1, 10, 4)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.caption("Abubakar Salawu · A00019166 · School of IT and Computing, AUN. "
               "Sentiment labels follow the NaijaSenti encoding 0 negative, 1 neutral, 2 positive.")


@st.cache_data(show_spinner=False)
def _load_folder(path: str, seed: int = 42):
    return core.load_bundle(path, allow_demo=True, seed=seed)


if mode == "Upload files":
    bundle = core.bundle_from_uploads({k: v for k, v in uploads.items() if v is not None})
elif mode == "Demonstration only":
    bundle = core.demo_bundle()
else:
    bundle = _load_folder(root)

clf, series, fc = bundle.classification, bundle.series, bundle.forecasts

# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1.15])
with head_l:
    st.markdown("<div class='eyebrow'>Jumia · Konga · Temu — daily</div>", unsafe_allow_html=True)
    st.markdown("<h1>Marketplace Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<div class='lede'>Code-mixed consumer sentiment, read a day before the price "
                "moves, and folded into a forecast that a price-only model cannot see.</div>",
                unsafe_allow_html=True)
with head_r:
    live = [k for k, v in bundle.origin.items() if v not in ("demonstration", "missing")]
    if bundle.is_demo and not live:
        st.markdown("<div class='chip demo'>◧ <b>Demonstration data.</b> No pipeline output "
                    "loaded — nothing here is a model result.</div>", unsafe_allow_html=True)
    elif bundle.is_demo:
        st.markdown(f"<div class='chip demo'>◧ <b>Mixed.</b> {len(live)} of 3 tables loaded from "
                    "the pipeline; the rest is demonstration data.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='chip'>◧ <b>Pipeline output.</b> All three result tables "
                    "loaded.</div>", unsafe_allow_html=True)

tab_overview, tab_rq1, tab_rq2, tab_rq3, tab_warn, tab_data = st.tabs(
    ["Overview", "Sentiment · RQ1", "Lead & lag · RQ2", "Forecast · RQ3",
     "Early warning", "Data & export"])

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    clf_tbl = core.classifier_table(clf)
    fc_tbl = core.forecast_table(fc)
    best_clf = clf_tbl.loc[clf_tbl["Macro F1"].idxmax()]
    fusion_row = fc_tbl[fc_tbl["arm"] == core.EXPERIMENTAL_ARM]
    cc = core.cross_correlation(series, max_lag=10)
    peak = cc.loc[cc["r"].abs().idxmax()]
    vol = core.realised_volatility(series)

    impr = float(fusion_row["Δ vs control %"].iloc[0]) if len(fusion_row) else float("nan")
    cards([
        card("Best sentiment model", f"{best_clf['Macro F1']:.3f}",
             f"macro F1 · {best_clf['Model']}", "var(--teal)"),
        card("Fusion vs price-only", f"{impr:+.1f}%",
             "change in mean absolute error against the LSTM control", "var(--plum)"),
        card("Strongest lead", f"{int(peak['lag']):+d} d",
             f"r = {peak['r']:+.2f} · {peak['direction']}", "var(--marigold)"),
        card("Realised volatility", f"{vol * 100:.1f}%",
             "standard deviation of daily returns, last 30 days", "var(--slate)"),
    ])

    section("Sentiment ahead of price",
            "Each cell is one day of consumer sentiment, shifted forward by a day and laid under "
            "the price it precedes. Runs of clay-coloured cells before a climb are the pattern the "
            "fusion model is built to exploit.")
    st.altair_chart(ch.signal_ribbon(series, days=window, lead=1), **_WIDE)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<h3>Where each model lands</h3>", unsafe_allow_html=True)
        st.altair_chart(ch.model_ranking(clf_tbl), **_WIDE)
    with right:
        st.markdown("<h3>Forecast error against the control</h3>", unsafe_allow_html=True)
        st.altair_chart(ch.improvement_bars(fc_tbl, core.FC_ARMS[core.CONTROL_ARM]),
                        **_WIDE)
    st.markdown("<div class='note'>Positive bars mean lower error than the price-only LSTM. "
                "The significance test for that gap is on the Forecast tab.</div>",
                unsafe_allow_html=True)

# --------------------------------------------------------------------------
# RQ1 — sentiment classification
# --------------------------------------------------------------------------
with tab_rq1:
    st.markdown("<div class='eyebrow'>Research question 1</div>"
                "<h2>Can language-adaptive fine-tuning read code-mixed Nigerian commerce talk?</h2>",
                unsafe_allow_html=True)

    clf_tbl = core.classifier_table(clf)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.altair_chart(ch.model_ranking(clf_tbl), **_WIDE)
        st.dataframe(
            clf_tbl.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}",
                                  "Weighted F1": "{:.3f}"}),
            **_WIDE, hide_index=True)
    with c2:
        arm_keys = [k for k in core.CLF_ARMS if k in clf.columns]
        pick = st.selectbox("Inspect a model", arm_keys,
                            index=len(arm_keys) - 1,
                            format_func=lambda k: core.CLF_ARMS[k])
        cm = core.confusion(clf["y_true"].to_numpy(), clf[pick].to_numpy())
        st.altair_chart(ch.confusion_heatmap(cm, core.LABELS), **_WIDE)
        pc = core.per_class_metrics(clf["y_true"].to_numpy(), clf[pick].to_numpy())
        st.dataframe(pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                     **_WIDE, hide_index=True)

    section("Is the gap real?",
            "McNemar compares two classifiers on the same test items, counting only the cases "
            "where they disagree.")
    m1, m2, m3 = st.columns([1, 1, 2])
    with m1:
        a = st.selectbox("Model A", arm_keys, index=0, format_func=lambda k: core.CLF_ARMS[k])
    with m2:
        b = st.selectbox("Model B", arm_keys, index=len(arm_keys) - 1,
                         format_func=lambda k: core.CLF_ARMS[k])
    res = core.mcnemar(clf["y_true"].to_numpy(), clf[a].to_numpy(), clf[b].to_numpy())
    with m3:
        verdict = ("no measurable difference" if np.isnan(res["p_value"]) or res["p_value"] >= 0.05
                   else "a difference beyond chance at the 5% level")
        cards([
            card("Discordant pairs", f"{res['n01'] + res['n10']:,}",
                 f"B right where A wrong: {res['n01']:,} · A right where B wrong: {res['n10']:,}",
                 "var(--slate)"),
            card("p-value", "—" if np.isnan(res["p_value"]) else f"{res['p_value']:.4f}",
                 f"χ² = {res['statistic']:.2f} · reads as {verdict}", "var(--teal)"),
        ])

# --------------------------------------------------------------------------
# RQ2 — lead and lag
# --------------------------------------------------------------------------
with tab_rq2:
    st.markdown("<div class='eyebrow'>Research question 2</div>"
                "<h2>Does sentiment move before price, or only alongside it?</h2>",
                unsafe_allow_html=True)

    st.altair_chart(ch.standardised_pair(series, days=window), **_WIDE)

    l, r = st.columns([1.1, 1])
    with l:
        st.markdown("<h3>Cross-correlation by lag</h3>", unsafe_allow_html=True)
        cc = core.cross_correlation(series, max_lag=10)
        st.altair_chart(ch.crosscorr(cc), **_WIDE)
    with r:
        st.markdown("<h3>Granger causality</h3>", unsafe_allow_html=True)
        g = core.granger_table(series, max_lag=max_lag)
        if g.empty:
            st.info("Granger testing needs statsmodels and at least a few weeks of paired "
                    "daily observations. Install statsmodels or widen the series.")
        else:
            st.altair_chart(ch.granger_dots(g), **_WIDE)
            best = g[g["Direction"] == "sentiment → price"].sort_values("p").head(1)
            if len(best):
                row = best.iloc[0]
                st.markdown(f"<div class='note'>Strongest evidence at lag {int(row['Lag (days)'])}: "
                            f"F = {row['F']:.2f}, p = {row['p']:.4f}. Both series are "
                            "first-differenced before testing.</div>", unsafe_allow_html=True)

    with st.expander("Stationarity checks (augmented Dickey–Fuller)"):
        adf = core.adf_summary(series)
        if adf.empty:
            st.write("statsmodels is not available in this environment.")
        else:
            st.dataframe(adf.style.format({"ADF": "{:.3f}", "p": "{:.4f}"}),
                         **_WIDE, hide_index=True)
        st.caption("Granger causality assumes stationary inputs, which is why the test above runs "
                   "on differences rather than levels.")

# --------------------------------------------------------------------------
# RQ3 — forecasting
# --------------------------------------------------------------------------
with tab_rq3:
    st.markdown("<div class='eyebrow'>Research question 3</div>"
                "<h2>Does fusing the two signals forecast better than price alone?</h2>",
                unsafe_allow_html=True)

    available = [a for a in core.FC_ARMS if a in fc.columns]
    picked = st.multiselect("Arms to plot", available,
                            default=[core.CONTROL_ARM, core.EXPERIMENTAL_ARM],
                            format_func=lambda k: core.FC_ARMS[k])
    st.altair_chart(ch.forecast_lines(fc, picked, core.FC_ARMS), **_WIDE)

    fc_tbl = core.forecast_table(fc)
    t1, t2 = st.columns([1.3, 1])
    with t1:
        st.dataframe(
            fc_tbl.drop(columns=["arm"]).style.format(
                {"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.2f}",
                 "Δ vs control %": "{:+.1f}"}),
            **_WIDE, hide_index=True)
        st.markdown("<div class='note'>Held-out window only. The control is the price-only LSTM, "
                    "so its own row reads zero by construction.</div>", unsafe_allow_html=True)
    with t2:
        dm = core.diebold_mariano(fc["y_true"], fc[core.CONTROL_ARM], fc[core.EXPERIMENTAL_ARM]) \
            if core.CONTROL_ARM in fc.columns and core.EXPERIMENTAL_ARM in fc.columns else \
            {"DM": float("nan"), "p_value": float("nan"), "n": 0}
        sign = ("fusion has the lower loss" if dm["DM"] > 0 else "the control has the lower loss")
        cards([
            card("Diebold–Mariano", "—" if np.isnan(dm["DM"]) else f"{dm['DM']:.2f}",
                 f"{sign} · n = {dm['n']}", "var(--plum)"),
            card("p-value", "—" if np.isnan(dm["p_value"]) else f"{dm['p_value']:.4f}",
                 "two-sided, absolute-loss differential, HLN-corrected", "var(--teal)"),
        ])

    section("Residuals over the test window",
            "Points above the line are days the model priced too low, below it too high. "
            "Drift or clustering here matters more than the headline average.")
    st.altair_chart(ch.residual_scatter(fc, picked or [core.EXPERIMENTAL_ARM], core.FC_ARMS),
                    **_WIDE)

# --------------------------------------------------------------------------
# Early warning
# --------------------------------------------------------------------------
with tab_warn:
    st.markdown("<div class='eyebrow'>Decision tool</div>"
                "<h2>What should a seller do tomorrow?</h2>"
                "<div class='lede'>Paste the comments you are seeing today. The panel scores them, "
                "reads them against the price relationship fitted on the loaded series, and returns "
                "a level.</div>", unsafe_allow_html=True)

    a, b = st.columns([1, 1.35])
    with a:
        product = st.text_input("Product or SKU", "5 kg rice")
        platform = st.selectbox("Marketplace", ["Jumia", "Konga", "Temu"])
        horizon = st.slider("Days ahead", 3, 21, 7)
        shift = st.slider("Scenario: shift sentiment by (standard deviations)",
                          -2.0, 2.0, -1.0, 0.25)
        comments_raw = st.text_area(
            "Consumer comments, one per line", height=170,
            value="This rice na correct, sweet well well\n"
                  "Price don too cost, wahala\n"
                  "Original product, fast delivery, e good\n"
                  "Last one I bought was fake, scam")
        run = st.button("Read the signal", **_WIDE)

    with b:
        if run:
            scored, net = core.score_comments(comments_raw.splitlines())
            vol = core.realised_volatility(series)
            risk = core.risk_assessment(net, vol)
            sens = core.sentiment_sensitivity(series, lag=1)
            proj = core.project_price(series, horizon, shift)

            st.markdown(
                f"<div class='warn'><div class='lvl {risk['tone']}'>{risk['level']}</div>"
                f"<div><div class='eyebrow'>{product} · {platform} · next {horizon} days</div>"
                f"<div class='txt'>{risk['note']}. Net sentiment {net:+.2f}, "
                f"realised volatility {vol * 100:.1f}%.</div></div></div>",
                unsafe_allow_html=True)

            if not proj.empty:
                delta = proj["Scenario"].iloc[-1] - proj["Baseline"].iloc[-1]
                cards([
                    card("Price if sentiment holds", f"₦{proj['Baseline'].iloc[-1]:,.0f}",
                         f"day {horizon}", "var(--slate)"),
                    card("Price under the scenario", f"₦{proj['Scenario'].iloc[-1]:,.0f}",
                         f"{delta:+,.0f} ₦ against the baseline", "var(--plum)"),
                    card("Sentiment sensitivity", f"₦{sens['beta']:,.0f}",
                         "price response to a one-SD sentiment move, one day later",
                         "var(--marigold)"),
                ])
                st.altair_chart(ch.scenario_chart(series, proj), **_WIDE)

            counts = (scored["Label"].value_counts().rename_axis("Label")
                      .reset_index(name="Count")) if not scored.empty else pd.DataFrame()
            if not counts.empty:
                st.altair_chart(ch.sentiment_mix(counts), **_WIDE)
                st.dataframe(scored, hide_index=True, **_WIDE)

            engine = ("the registered AfriBERTa + LAFT classifier"
                      if core.classifier_is_registered()
                      else "a small code-mixed keyword lexicon, not the trained classifier")
            st.markdown(f"<div class='note'>Comments were labelled with {engine}. The projection "
                        "rolls forward an AR(1) fit with lagged sentiment, estimated on the series "
                        "currently loaded — it is a scenario, not the fusion model's forecast.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn'><div class='lvl steady'>—</div><div>"
                        "<div class='eyebrow'>Nothing read yet</div>"
                        "<div class='txt'>Enter today's comments on the left and select "
                        "<b>Read the signal</b>.</div></div></div>", unsafe_allow_html=True)

    with st.expander("How to plug the trained classifier in"):
        st.code(
            "# in app.py, just below `import core`\n"
            "from transformers import pipeline\n\n"
            "pipe = pipeline('text-classification', model='models/laft_cls',\n"
            "                tokenizer='models/laft_cls', truncation=True, max_length=128)\n"
            "LMAP = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}\n\n"
            "core.register_classifier(lambda texts: [LMAP[r['label']] for r in pipe(texts)])",
            language="python")
        st.caption("Once registered, this panel labels with the model and the lexicon note "
                   "disappears.")

# --------------------------------------------------------------------------
# Data and export
# --------------------------------------------------------------------------
with tab_data:
    st.markdown("<div class='eyebrow'>Provenance</div><h2>What is loaded right now</h2>",
                unsafe_allow_html=True)
    st.dataframe(bundle.status(), hide_index=True, **_WIDE)

    section("Tables for Chapter 6", "Exactly the numbers shown above, ready to paste.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Classification results (CSV)",
                       core.classifier_table(clf).to_csv(index=False),
                       "table_6_3_classification.csv", **_WIDE)
    d2.download_button("Forecasting results (CSV)",
                       core.forecast_table(fc).drop(columns=["arm"]).to_csv(index=False),
                       "table_6_4_forecasting.csv", **_WIDE)
    g = core.granger_table(series, max_lag=max_lag)
    d3.download_button("Granger results (CSV)",
                       (g if not g.empty else pd.DataFrame({"note": ["unavailable"]})).to_csv(index=False),
                       "table_6_granger.csv", **_WIDE)

    section("The series behind the charts")
    st.dataframe(series.tail(200), hide_index=True, **_WIDE)

    st.markdown("<div class='note'>Expected schema — <b>classification.csv</b>: y_true, "
                "b1_svm_tfidf, b2_transformer_nolaft, laft_afriberta · "
                "<b>series_daily.csv</b>: date, price, sentiment · "
                "<b>forecasts.csv</b>: date, y_true, arima, xgboost, lstm_price, sentiment_only, "
                "fusion.</div>", unsafe_allow_html=True)
