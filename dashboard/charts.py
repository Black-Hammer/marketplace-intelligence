"""
charts.py — every chart in the dashboard, on one shared Altair theme.

Colour roles are fixed across the whole application so a reader learns them once:
    teal      price / actual values
    marigold  sentiment
    plum      the fusion (experimental) arm
    slate     baselines and controls
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

INK = "#12211F"
MUTED = "#5F726D"
LINE = "#D7DEDB"
PAPER = "#F3F5F4"

TEAL = "#0E6B60"
MARIGOLD = "#C98A0E"
PLUM = "#6B3E7A"
SLATE = "#7C8D95"
CLAY = "#A4271C"

DISPLAY = "Bricolage Grotesque, Georgia, serif"
BODY = "IBM Plex Sans, Helvetica Neue, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"

ARM_COLOURS = {
    "y_true": INK,
    "Actual": INK,
    "arima": "#9AA7AE",
    "xgboost": SLATE,
    "lstm_price": TEAL,
    "sentiment_only": MARIGOLD,
    "fusion": PLUM,
}


def _theme() -> dict:
    return {
        "config": {
            "background": "transparent",
            "font": BODY,
            "view": {"stroke": "transparent", "continuousHeight": 260},
            "axis": {
                "labelFont": MONO, "labelFontSize": 10, "labelColor": MUTED,
                "titleFont": MONO, "titleFontSize": 10, "titleColor": MUTED,
                "titleFontWeight": "normal", "titlePadding": 10,
                "domainColor": LINE, "tickColor": LINE, "gridColor": "#E7ECEA",
                "labelPadding": 6,
            },
            "legend": {
                "labelFont": BODY, "labelFontSize": 11, "labelColor": INK,
                "titleFont": MONO, "titleFontSize": 10, "titleColor": MUTED,
                "titleFontWeight": "normal", "symbolType": "stroke", "symbolStrokeWidth": 3,
                "orient": "top", "direction": "horizontal", "offset": 4,
            },
            "title": {"font": DISPLAY, "fontSize": 13, "color": INK, "anchor": "start",
                      "fontWeight": 600, "offset": 12},
            "range": {"category": [TEAL, MARIGOLD, PLUM, SLATE, CLAY, "#9AA7AE"]},
        }
    }


def install_theme() -> None:
    """Register the theme across Altair 4/5/6 APIs."""
    try:                                    # Altair >= 5.5
        alt.theme.register("marketplace", enable=True)(_theme)
        alt.theme.enable("marketplace")
        return
    except Exception:
        pass
    try:                                    # Altair 4 / early 5
        alt.themes.register("marketplace", _theme)
        alt.themes.enable("marketplace")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Signature: the signal ribbon
# --------------------------------------------------------------------------

def signal_ribbon(series: pd.DataFrame, days: int = 120, lead: int = 1) -> alt.VConcatChart:
    """Daily sentiment as a ribbon of cells, with the price path riding above it.

    The ribbon is shifted right by `lead` days so a reader can see each day's
    sentiment sitting directly under the price it precedes.
    """
    d = series.tail(days).copy()
    d["cell_start"] = d["date"] + pd.Timedelta(days=lead)
    d["cell_end"] = d["cell_start"] + pd.Timedelta(days=1)

    span = alt.Scale(domain=[d["date"].min().isoformat(),
                             (d["date"].max() + pd.Timedelta(days=lead + 1)).isoformat()])
    axis = alt.Axis(format="%d %b", tickCount=6)

    price = (
        alt.Chart(d)
        .mark_line(color=TEAL, strokeWidth=1.8, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None, scale=span, axis=axis),
            y=alt.Y("price:Q", title="price (NGN)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("date:T", title="Day"),
                     alt.Tooltip("price:Q", title="Price", format=",.0f")],
        )
        .properties(height=210)
    )

    ribbon = (
        alt.Chart(d)
        .mark_rect(stroke=PAPER, strokeWidth=0.5)
        .encode(
            x=alt.X("cell_start:T", title=None, scale=span, axis=axis),
            x2="cell_end:T",
            color=alt.Color(
                "sentiment:Q",
                scale=alt.Scale(domain=[-2.5, 0, 2.5], range=[CLAY, "#F0EAD8", TEAL]),
                legend=alt.Legend(title=f"sentiment, shifted +{lead}d", gradientLength=140,
                                  orient="top", direction="horizontal"),
            ),
            tooltip=[alt.Tooltip("date:T", title="Comment day"),
                     alt.Tooltip("sentiment:Q", title="Sentiment (z)", format=".2f")],
        )
        .properties(height=26)
    )
    return alt.vconcat(price, ribbon, spacing=4).resolve_scale(color="independent")


# --------------------------------------------------------------------------
# RQ1 — classification
# --------------------------------------------------------------------------

def model_ranking(table: pd.DataFrame, metric: str = "Macro F1") -> alt.LayerChart:
    base = alt.Chart(table).encode(
        y=alt.Y("Model:N", sort="-x", title=None),
        x=alt.X(f"{metric}:Q", title=metric.lower(), scale=alt.Scale(domain=[0, 1])),
    )
    bars = base.mark_bar(height=16, cornerRadiusEnd=3, color=TEAL)
    labels = base.mark_text(align="left", dx=6, font=MONO, fontSize=10, color=INK).encode(
        text=alt.Text(f"{metric}:Q", format=".3f"))
    return (bars + labels).properties(height=alt.Step(34))


def confusion_heatmap(cm: np.ndarray, labels: dict) -> alt.LayerChart:
    rows = [{"Actual": labels[i], "Predicted": labels[j], "n": int(cm[i, j]),
             "share": float(cm[i, j] / cm[i].sum()) if cm[i].sum() else 0.0}
            for i in labels for j in labels]
    d = pd.DataFrame(rows)
    order = list(labels.values())
    base = alt.Chart(d).encode(
        x=alt.X("Predicted:N", sort=order, title="predicted"),
        y=alt.Y("Actual:N", sort=order, title="actual"),
    )
    cells = base.mark_rect(cornerRadius=2).encode(
        color=alt.Color("share:Q", scale=alt.Scale(scheme="teals", domain=[0, 1]),
                        legend=alt.Legend(title="row share", format=".0%")),
        tooltip=["Actual", "Predicted", alt.Tooltip("n:Q", title="Count"),
                 alt.Tooltip("share:Q", format=".1%", title="Row share")],
    )
    text = base.mark_text(font=MONO, fontSize=11).encode(
        text=alt.Text("n:Q", format=","),
        color=alt.condition(alt.datum.share > 0.5, alt.value("white"), alt.value(INK)),
    )
    return (cells + text).properties(height=200)


# --------------------------------------------------------------------------
# RQ2 — lead and lag
# --------------------------------------------------------------------------

def standardised_pair(series: pd.DataFrame, days: int = 180) -> alt.Chart:
    d = series.tail(days).copy()
    z = lambda x: (x - x.mean()) / (x.std() or 1)
    long = pd.concat([
        pd.DataFrame({"date": d["date"], "value": z(d["price"]), "Signal": "price (z)"}),
        pd.DataFrame({"date": d["date"], "value": z(d["sentiment"]), "Signal": "sentiment (z)"}),
    ])
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1.6, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("value:Q", title="standardised"),
            color=alt.Color("Signal:N", scale=alt.Scale(domain=["price (z)", "sentiment (z)"],
                                                        range=[TEAL, MARIGOLD]), title=None,
                            legend=alt.Legend(orient="top", direction="horizontal")),
            tooltip=["date:T", "Signal:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=240)
    )


def crosscorr(cc: pd.DataFrame) -> alt.LayerChart:
    peak = cc.loc[cc["r"].abs().idxmax()]
    bars = (
        alt.Chart(cc)
        .mark_bar(size=12, cornerRadius=2)
        .encode(
            x=alt.X("lag:O", title="lag in days   ← price leads · sentiment leads →"),
            y=alt.Y("r:Q", title="correlation"),
            color=alt.condition(alt.datum.lag > 0, alt.value(MARIGOLD), alt.value(SLATE)),
            tooltip=[alt.Tooltip("lag:O", title="Lag"), alt.Tooltip("r:Q", format=".3f"),
                     alt.Tooltip("direction:N", title="Reading")],
        )
    )
    dy = 14 if float(peak["r"]) < 0 else -14
    marker = (
        alt.Chart(pd.DataFrame([{"lag": int(peak["lag"]), "r": float(peak["r"])}]))
        .mark_text(text="strongest", dy=dy, font=MONO, fontSize=10, color=INK)
        .encode(x="lag:O", y="r:Q")
    )
    return (bars + marker).properties(height=230)


def granger_dots(g: pd.DataFrame) -> alt.LayerChart:
    rule = alt.Chart(pd.DataFrame({"p": [0.05]})).mark_rule(
        color=CLAY, strokeDash=[4, 3], strokeWidth=1).encode(y="p:Q")
    dots = (
        alt.Chart(g)
        .mark_circle(size=110, opacity=0.9)
        .encode(
            x=alt.X("Lag (days):O", title="lag"),
            y=alt.Y("p:Q", title="p-value", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Direction:N", scale=alt.Scale(
                domain=["sentiment → price", "price → sentiment"], range=[MARIGOLD, SLATE]), title=None),
            tooltip=["Direction:N", "Lag (days):O", alt.Tooltip("F:Q", format=".2f"),
                     alt.Tooltip("p:Q", format=".4f")],
        )
    )
    return (dots + rule).properties(height=230)


# --------------------------------------------------------------------------
# RQ3 — forecasting
# --------------------------------------------------------------------------

def forecast_lines(fc: pd.DataFrame, arms: list[str], arm_labels: dict) -> alt.Chart:
    frames = [pd.DataFrame({"date": fc["date"], "value": fc["y_true"], "Series": "Actual"})]
    for a in arms:
        if a in fc.columns:
            frames.append(pd.DataFrame({"date": fc["date"], "value": fc[a],
                                        "Series": arm_labels.get(a, a)}))
    long = pd.concat(frames)
    domain = ["Actual"] + [arm_labels.get(a, a) for a in arms if a in fc.columns]
    rng = [INK] + [ARM_COLOURS.get(a, SLATE) for a in arms if a in fc.columns]
    dash = alt.Scale(domain=domain, range=[[1, 0]] + [[1, 0]] * (len(domain) - 1))
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1.7, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("value:Q", title="price (₦)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=rng), title=None,
                            legend=alt.Legend(orient="top", direction="horizontal")),
            strokeDash=alt.StrokeDash("Series:N", scale=dash, legend=None),
            opacity=alt.condition(alt.datum.Series == "Actual", alt.value(1.0), alt.value(0.85)),
            tooltip=["date:T", "Series:N", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=300)
    )


def improvement_bars(table: pd.DataFrame, control_label: str) -> alt.LayerChart:
    d = table[table["Model"] != control_label].copy()
    base = alt.Chart(d).encode(
        y=alt.Y("Model:N", sort="-x", title=None),
        x=alt.X("Δ vs control %:Q", title="MAE change against the price-only control (%)"),
    )
    bars = base.mark_bar(height=16, cornerRadiusEnd=3).encode(
        color=alt.condition(alt.datum["Δ vs control %"] > 0, alt.value(PLUM), alt.value(SLATE)))
    labels = base.mark_text(align="left", dx=6, font=MONO, fontSize=10, color=INK).encode(
        text=alt.Text("Δ vs control %:Q", format="+.1f"))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=LINE).encode(x="x:Q")
    return (zero + bars + labels).properties(height=alt.Step(32))


def residual_scatter(fc: pd.DataFrame, arms: list[str], arm_labels: dict) -> alt.Chart:
    frames = []
    for a in arms:
        if a in fc.columns:
            frames.append(pd.DataFrame({
                "date": fc["date"], "residual": fc["y_true"] - fc[a],
                "Series": arm_labels.get(a, a)}))
    if not frames:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()
    long = pd.concat(frames)
    domain = sorted(long["Series"].unique())
    rng = [ARM_COLOURS.get(k, SLATE) for a in arms if a in fc.columns
           for k in [a]][:len(domain)]
    return (
        alt.Chart(long)
        .mark_circle(size=45, opacity=0.7)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("residual:Q", title="actual − predicted (₦)", axis=alt.Axis(format=",.0f")),
            color=alt.Color("Series:N", scale=alt.Scale(range=rng), title=None),
            tooltip=["date:T", "Series:N", alt.Tooltip("residual:Q", format=",.0f")],
        )
        .properties(height=230)
    )


# --------------------------------------------------------------------------
# Early warning
# --------------------------------------------------------------------------

def scenario_chart(history: pd.DataFrame, projection: pd.DataFrame, tail: int = 60) -> alt.LayerChart:
    h = history.tail(tail)[["date", "price"]].rename(columns={"price": "value"})
    h["Series"] = "Observed"
    frames = [h]
    for col, name in (("Baseline", "Sentiment holds"), ("Scenario", "Scenario")):
        if col in projection.columns:
            frames.append(pd.DataFrame({"date": projection["date"],
                                        "value": projection[col], "Series": name}))
    long = pd.concat(frames)
    domain = ["Observed", "Sentiment holds", "Scenario"]
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.9, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("value:Q", title="price (₦)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain,
                            range=[INK, SLATE, PLUM]), title=None),
            strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
                domain=domain, range=[[1, 0], [4, 3], [1, 0]]), legend=None),
            tooltip=["date:T", "Series:N", alt.Tooltip("value:Q", format=",.0f")],
        )
    )
    split = alt.Chart(pd.DataFrame({"date": [history["date"].iloc[-1]]})).mark_rule(
        color=LINE, strokeWidth=1).encode(x="date:T")
    return (split + lines).properties(height=280)


def sentiment_mix(counts: pd.DataFrame) -> alt.LayerChart:
    base = alt.Chart(counts).encode(
        y=alt.Y("Label:N", sort=["Positive", "Neutral", "Negative"], title=None),
        x=alt.X("Count:Q", title="comments"),
    )
    bars = base.mark_bar(height=18, cornerRadiusEnd=3).encode(
        color=alt.Color("Label:N", scale=alt.Scale(
            domain=["Positive", "Neutral", "Negative"], range=[TEAL, SLATE, CLAY]), legend=None))
    labels = base.mark_text(align="left", dx=6, font=MONO, fontSize=10, color=INK).encode(
        text="Count:Q")
    return (bars + labels).properties(height=alt.Step(30))
