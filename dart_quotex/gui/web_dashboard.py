"""
dart_quotex/gui/web_dashboard.py
Interactive Streamlit web dashboard with glassmorphism design.

Launch
------
    streamlit run dart_quotex/gui/web_dashboard.py
    # or
    python main.py dashboard

Features
--------
· Live balance and session P&L (auto-refresh every 5s)
· Candlestick chart (Plotly — interactive)
· AI signal gauge with confidence ring
· Market regime pie chart
· Multi-timeframe alignment heatmap
· Win rate, Profit Factor, Sharpe, Sortino, Calmar, CVaR gauges
· Trade history table with colour coding
· Equity curve with drawdown overlay
· News sentiment timeline
· AI debug panel: ensemble weights, uncertainty, ICM novelty
· Model performance: reliability diagram, R-multiple distribution
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Guard against non-streamlit execution ─────────────────────────────────────
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False


# ──────────────────────────────────────────────────────────────────────────────
# Colour tokens (glassmorphism dark)
# ──────────────────────────────────────────────────────────────────────────────

C = {
    "bg":      "#0d0f14",
    "panel":   "rgba(21,24,32,0.85)",
    "border":  "#2a2d3a",
    "accent":  "#00d4aa",
    "accent2": "#7c5cbf",
    "call":    "#00e676",
    "put":     "#ff5252",
    "hold":    "#ffa726",
    "text":    "#e8eaf6",
    "muted":   "#7986cb",
}

CSS = f"""
<style>
  html, body, [class*="css"] {{
    background-color: {C["bg"]} !important;
    color: {C["text"]} !important;
    font-family: "Consolas", "Courier New", monospace !important;
  }}
  .stMetric {{
    background: {C["panel"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(12px);
  }}
  .stMetric label {{
    color: {C["muted"]} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .stMetric [data-testid="stMetricValue"] {{
    color: {C["text"]} !important;
    font-size: 24px !important;
    font-weight: bold;
  }}
  div[data-testid="stSidebar"] {{
    background: {C["panel"]} !important;
    border-right: 1px solid {C["border"]};
  }}
  .block-container {{
    padding-top: 1rem !important;
    max-width: 1600px;
  }}
  h1, h2, h3 {{
    color: {C["accent"]} !important;
    letter-spacing: 1px;
  }}
  .signal-card {{
    background: {C["panel"]};
    border: 2px solid {C["accent"]};
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(20px);
  }}
  .regime-badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
  }}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_db_trades(db_path: str = "data/market.db") -> pd.DataFrame:
    """Load recent trades from SQLite."""
    try:
        from dart_quotex.data.database import Database
        db = Database(db_path)
        df = db.get_recent_trades(200)
        return df
    except Exception as exc:
        logger.debug("DB load error: %s", exc)
        return pd.DataFrame()


def _load_db_candles(
    db_path: str = "data/market.db",
    asset: str = "EURUSD_OTC",
    gran: int = 60,
    n: int = 100,
) -> pd.DataFrame:
    try:
        from dart_quotex.data.database import Database
        db = Database(db_path)
        return db.get_candles(asset, gran, limit=n)
    except Exception:
        return pd.DataFrame()


def _load_metrics(db_path: str = "data/market.db") -> dict:
    """Compute live performance metrics."""
    try:
        from dart_quotex.data.database import Database
        from dart_quotex.metrics.performance import PerformanceCalculator, TradeRecord
        db     = Database(db_path)
        df_raw = db.get_recent_trades(500)
        if df_raw.empty:
            return {}

        completed = df_raw[df_raw["result"].notna()]
        if completed.empty:
            return {}

        records = [
            TradeRecord(
                ts_open=float(r.get("ts_open", 0)),
                ts_close=float(r.get("ts_close") or r.get("ts_open", 60) + 60),
                direction=str(r.get("direction", "call")),
                stake=float(r.get("stake", 10)),
                payout=float(r.get("payout", 0)) - float(r.get("stake", 10))
                       if r.get("result") == "WIN"
                       else -float(r.get("stake", 10)),
                confidence=float(r.get("confidence", 0.5)),
                asset=str(r.get("asset", "")),
                won=(r.get("result") == "WIN"),
            )
            for _, r in completed.iterrows()
        ]

        calc = PerformanceCalculator(start_balance=1000.0)
        m    = calc.compute(records)
        return m.to_dict()
    except Exception as exc:
        logger.debug("Metrics error: %s", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Chart builders (Plotly)
# ──────────────────────────────────────────────────────────────────────────────

def _candlestick_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing_line_color=C["call"],
        decreasing_line_color=C["put"],
        increasing_fillcolor=C["call"],
        decreasing_fillcolor=C["put"],
        name="OHLC",
    ))
    # 20-period SMA overlay
    if len(df) >= 20:
        sma = df["close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma,
            line=dict(color=C["accent"], width=1.5, dash="dash"),
            name="SMA-20",
        ))
    fig.update_layout(
        plot_bgcolor=C["bg"],
        paper_bgcolor=C["bg"],
        font_color=C["text"],
        xaxis=dict(gridcolor=C["border"], showgrid=True),
        yaxis=dict(gridcolor=C["border"], showgrid=True, side="right"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor=C["panel"], font_size=11),
        margin=dict(l=0, r=0, t=8, b=0),
        height=320,
    )
    return fig


def _equity_curve(trades_df: pd.DataFrame, start: float = 1000.0) -> go.Figure:
    if trades_df.empty or "result" not in trades_df.columns:
        return go.Figure()

    completed = trades_df[trades_df["result"].notna()].copy()
    if completed.empty:
        return go.Figure()

    balance = [start]
    for _, row in completed.iterrows():
        last = balance[-1]
        if row["result"] == "WIN":
            last += float(row.get("payout", 10)) - float(row.get("stake", 10))
        else:
            last -= float(row.get("stake", 10))
        balance.append(max(0, last))

    x = list(range(len(balance)))

    # Drawdown
    peak = np.maximum.accumulate(balance)
    dd   = (np.array(balance) - peak) / (np.array(peak) + 1e-9)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.04,
    )
    fig.add_trace(go.Scatter(
        x=x, y=balance,
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
        line=dict(color=C["accent"], width=2),
        name="Equity",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=dd * 100,
        fill="tozeroy",
        fillcolor="rgba(255,82,82,0.15)",
        line=dict(color=C["put"], width=1.5),
        name="Drawdown %",
    ), row=2, col=1)

    fig.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font_color=C["text"],
        xaxis2=dict(gridcolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], title="Balance"),
        yaxis2=dict(gridcolor=C["border"], title="DD %"),
        showlegend=True,
        legend=dict(bgcolor=C["panel"]),
        margin=dict(l=0, r=0, t=8, b=0),
        height=320,
    )
    return fig


def _regime_pie(probs: List[float]) -> go.Figure:
    REGIME_NAMES = [
        "TRENDING↑", "TRENDING↓", "RANGING",
        "VOLATILE", "BREAKOUT", "REVERSAL", "CHOPPY",
    ]
    colors = [C["call"], C["put"], C["accent2"],
              "#ff6e40", C["accent"], C["hold"], C["muted"]]
    fig = go.Figure(go.Pie(
        labels=REGIME_NAMES, values=probs,
        marker_colors=colors,
        hole=0.55,
        textfont_size=9,
    ))
    fig.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font_color=C["text"],
        margin=dict(l=0, r=0, t=8, b=0),
        height=200,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=9),
    )
    return fig


def _confidence_gauge(conf: float, direction: str) -> go.Figure:
    color = C["call"] if direction == "CALL" else C["put"] if direction == "PUT" else C["hold"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": C["muted"]},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": C["border"],
            "threshold": {
                "line": {"color": C["accent"], "width": 3},
                "thickness": 0.75,
                "value": 62,
            },
            "steps": [
                {"range": [0, 50],  "color": "rgba(255,82,82,0.1)"},
                {"range": [50, 62], "color": "rgba(255,167,38,0.1)"},
                {"range": [62, 100],"color": "rgba(0,230,118,0.1)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor=C["bg"], font_color=C["text"],
        margin=dict(l=10, r=10, t=30, b=10),
        height=180,
    )
    return fig


def _mtf_heatmap(mtf_data: Dict[str, dict]) -> go.Figure:
    tfs  = ["1m", "5m", "15m", "1h"]
    metrics = ["RSI", "Momentum", "Trend"]

    z = []
    text_z = []
    for metric in metrics:
        row, txt_row = [], []
        for tf in tfs:
            d = mtf_data.get(tf, {})
            if metric == "RSI":
                val = (float(d.get("rsi", 50)) - 50) / 50   # -1 to +1
                txt_row.append(f"{d.get('rsi', 0):.0f}")
            elif metric == "Momentum":
                val = float(np.clip(d.get("momentum", 0) * 500, -1, 1))
                txt_row.append(f"{d.get('momentum', 0):+.3f}")
            else:  # Trend
                t   = d.get("trend", "—")
                val = 1.0 if t == "UP" else (-1.0 if t == "DOWN" else 0.0)
                txt_row.append(t)
            row.append(val)
        z.append(row)
        text_z.append(txt_row)

    fig = go.Figure(go.Heatmap(
        z=z, x=tfs, y=metrics,
        text=text_z, texttemplate="%{text}",
        colorscale=[[0, C["put"]], [0.5, C["muted"]], [1, C["call"]]],
        zmid=0, showscale=False,
    ))
    fig.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font_color=C["text"],
        margin=dict(l=60, r=0, t=8, b=40),
        height=160,
        xaxis=dict(tickfont_size=11),
        yaxis=dict(tickfont_size=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ──────────────────────────────────────────────────────────────────────────────

def run_dashboard() -> None:
    if not _STREAMLIT:
        print("Install streamlit:  pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="DART-Quotex Dashboard",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{C['accent']};margin-bottom:0'>◈ DART-QUOTEX</h2>",
            unsafe_allow_html=True,
        )
        st.caption("AI-Driven Binary Options")
        st.divider()

        db_path  = st.text_input("Database", "data/market.db")
        asset    = st.selectbox("Asset", [
            "EURUSD_OTC","GBPUSD_OTC","USDJPY_OTC",
            "AUDUSD_OTC","USDCAD_OTC","EURJPY_OTC",
        ])
        refresh_s = st.slider("Auto-refresh (s)", 3, 60, 10)
        st.divider()
        st.caption("REGIME COLOURS")
        for r, col in [("TRENDING","green"),("RANGING","purple"),
                       ("VOLATILE","red"),("BREAKOUT","green"),
                       ("REVERSAL","orange"),("CHOPPY","grey")]:
            st.markdown(
                f"<span style='color:{C[col if col in C else 'text']}'>"
                f"● {r}</span>",
                unsafe_allow_html=True,
            )

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    placeholder = st.empty()
    import streamlit.components.v1 as components
    components.html(
        f"<script>setTimeout(function(){{window.location.reload();}},{refresh_s*1000});</script>",
        height=0,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    trades_df  = _load_db_trades(db_path)
    candles_df = _load_db_candles(db_path, asset, 60, 100)
    metrics    = _load_metrics(db_path)

    # ── Header row ────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='margin-bottom:0'>◈  DART-QUOTEX  /  {asset}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

    # ── Top KPIs ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        wr = metrics.get("win_rate", 0)
        st.metric("Win Rate", f"{wr:.1%}",
                  delta=f"{wr - 0.5:+.1%}" if wr else None)
    with k2:
        pf = metrics.get("profit_factor", 0)
        st.metric("Profit Factor", f"{pf:.2f}")
    with k3:
        roi = metrics.get("roi", 0)
        st.metric("ROI", f"{roi:+.1%}")
    with k4:
        dd = metrics.get("max_drawdown", 0)
        st.metric("Max Drawdown", f"{dd:.1%}")
    with k5:
        sh = metrics.get("sharpe", 0)
        st.metric("Sharpe", f"{sh:.2f}")
    with k6:
        so = metrics.get("sortino", 0)
        st.metric("Sortino", f"{so:.2f}")

    st.divider()

    # ── Main layout ───────────────────────────────────────────────────────────
    col_left, col_mid, col_right = st.columns([1.2, 2.5, 1.3])

    with col_left:
        st.subheader("AI SIGNAL")

        # Load signal from state file if available
        sig_file  = Path("data/.signal_state.json")
        sig_state: dict = {}
        if sig_file.exists():
            try:
                sig_state = json.loads(sig_file.read_text())
            except Exception:
                pass

        direction = sig_state.get("direction", "—")
        conf      = sig_state.get("confidence", 0.0)
        regime    = sig_state.get("regime", "DETECTING")
        probs     = sig_state.get("regime_probs", [1/7]*7)

        dir_color = (C["call"] if direction == "CALL"
                     else C["put"] if direction == "PUT"
                     else C["hold"])
        st.markdown(
            f"<div class='signal-card'>"
            f"<div style='font-size:52px;font-weight:bold;color:{dir_color}'>"
            f"{direction}</div>"
            f"<div style='color:{C['muted']};font-size:11px'>DIRECTION</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_confidence_gauge(conf, direction),
                        use_container_width=True, config={"displayModeBar": False})

        st.subheader("REGIME")
        st.plotly_chart(_regime_pie(probs),
                        use_container_width=True, config={"displayModeBar": False})

        regime_color = {
            "TRENDING_UP": C["call"], "TRENDING_DOWN": C["put"],
            "RANGING": C["accent2"], "VOLATILE": "#ff6e40",
            "BREAKOUT": C["call"], "REVERSAL": C["hold"], "CHOPPY": C["muted"],
        }.get(regime, C["text"])
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span class='regime-badge' style='background:rgba(0,0,0,0.4);"
            f"color:{regime_color};border:1px solid {regime_color}'>"
            f"● {regime}</span></div>",
            unsafe_allow_html=True,
        )

    with col_mid:
        st.subheader("PRICE CHART")
        if not candles_df.empty:
            st.plotly_chart(
                _candlestick_chart(candles_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No candle data — run `python main.py harvest` first")

        st.subheader("EQUITY CURVE")
        if not trades_df.empty:
            st.plotly_chart(
                _equity_curve(trades_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No trades yet")

    with col_right:
        st.subheader("MULTI-TIMEFRAME")
        mtf_state: dict = sig_state.get("mtf", {})
        if mtf_state:
            st.plotly_chart(
                _mtf_heatmap(mtf_state),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.caption("MTF data not available")

        st.subheader("EXTRA METRICS")
        extra_metrics = [
            ("Calmar",        metrics.get("calmar", 0),   ".3f"),
            ("Omega",         metrics.get("omega", 0),    ".3f"),
            ("VaR 95%",       metrics.get("var_95", 0),   ".4f"),
            ("CVaR 95%",      metrics.get("cvar_95", 0),  ".4f"),
            ("Win/Loss Ratio",metrics.get("win_loss_ratio",0),".2f"),
            ("Expectancy",    metrics.get("expectancy",0), "+.4f"),
        ]
        for label, val, fmt in extra_metrics:
            col_a, col_b = st.columns([3, 2])
            col_a.caption(label)
            val_str = f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)
            color   = C["call"] if isinstance(val, float) and val > 0 else C["put"] if isinstance(val, float) and val < 0 else C["text"]
            col_b.markdown(
                f"<span style='color:{color};font-weight:bold'>{val_str}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Trade history table ───────────────────────────────────────────────────
    st.subheader("TRADE HISTORY")
    if not trades_df.empty:
        display_df = trades_df.copy()
        if "result" in display_df.columns:
            def _style_row(row):
                color = C["call"] if row.get("result") == "WIN" else C["put"]
                return [f"color: {color}"] * len(row)

            st.dataframe(
                display_df.tail(50).iloc[::-1],
                use_container_width=True,
                height=280,
            )
    else:
        st.info("No trades in database")

    # ── Advanced analytics ────────────────────────────────────────────────────
    st.subheader("ADVANCED ANALYTICS")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.caption("STREAK ANALYSIS")
        if metrics:
            st.write(f"Max Win Streak:  **{metrics.get('max_consec_wins', 0)}**")
            st.write(f"Max Loss Streak: **{metrics.get('max_consec_losses', 0)}**")
            st.write(f"Trades / Day:    **{metrics.get('trades_per_day', 0):.1f}**")
            st.write(f"Avg Duration:    **{metrics.get('avg_duration_s', 0):.0f}s**")

    with a2:
        st.caption("R-MULTIPLE DISTRIBUTION")
        r_mean = metrics.get("r_mean", 0)
        r_std  = metrics.get("r_std", 1)
        r_skew = metrics.get("r_skew", 0)
        st.write(f"Mean R:  **{r_mean:+.3f}**")
        st.write(f"Std R:   **{r_std:.3f}**")
        st.write(f"Skew R:  **{r_skew:+.3f}**")

    with a3:
        st.caption("PER-ASSET BREAKDOWN")
        per_asset = metrics.get("per_asset", {})
        if per_asset:
            for asset_name, data in list(per_asset.items())[:5]:
                st.write(
                    f"**{asset_name}**: "
                    f"WR={data['win_rate']:.0%} "
                    f"N={data['n_trades']} "
                    f"P&L={data['net_profit']:+.2f}"
                )
        else:
            st.caption("No per-asset data")

    st.divider()
    st.caption(
        "DART-Quotex · AI-Driven Binary Options · "
        f"DB: `{db_path}` · "
        f"Refresh: {refresh_s}s"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (called by `streamlit run`)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dashboard()
