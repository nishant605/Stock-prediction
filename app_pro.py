"""
app_pro.py — StockVision AI Professional Dashboard
====================================================
Run with:  streamlit run app_pro.py
Original app.py and pages/predict.py are NOT modified.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prediction_engine import (
    load_data, get_available_stocks, get_display_name,
    get_sector, create_features, load_stock_model,
    predict_next_day, predict_multiple_days, get_backtest_results,
    get_stock_summary, STOCK_SECTORS
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StockVision AI — NIFTY50 Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
.stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 24px; padding: 3rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
    border: 1px solid rgba(102,126,234,0.15);
}
.hero::before {
    content: ''; position: absolute; top: -60%; right: -15%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(102,126,234,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: ''; position: absolute; bottom: -40%; left: -10%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(118,75,162,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; padding: 0.35rem 1.1rem; border-radius: 50px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem; position: relative;
}
.hero-sub {
    font-size: 1.1rem; color: #94a3b8; font-weight: 300;
    line-height: 1.7; max-width: 620px; position: relative;
}

/* ── Glassmorphism Card ── */
.g-card {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.4rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
}
.g-card:hover {
    border-color: rgba(102,126,234,0.35);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(102,126,234,0.12);
}
.g-card .label {
    font-size: 0.7rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 1.8px; font-weight: 700; margin-bottom: 0.5rem;
}
.g-card .value {
    font-size: 1.85rem; font-weight: 800; color: #e2e8f0;
    line-height: 1.2;
}
.val-green { color: #34d399 !important; }
.val-red { color: #f87171 !important; }
.val-blue { color: #60a5fa !important; }
.val-purple { color: #a78bfa !important; }
.val-amber { color: #fbbf24 !important; }

/* ── Pill ── */
.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.85rem; border-radius: 50px;
    font-size: 0.82rem; font-weight: 700; margin-top: 0.4rem;
}
.pill-up { background: rgba(52,211,153,0.12); color: #34d399; }
.pill-dn { background: rgba(248,113,113,0.12); color: #f87171; }

/* ── Section Title ── */
.sec-title {
    font-size: 1.3rem; font-weight: 800; color: #e2e8f0;
    margin: 2.2rem 0 1rem 0; padding-bottom: 0.6rem;
    border-bottom: 2px solid rgba(102,126,234,0.25);
    letter-spacing: -0.3px;
}

/* ── Prediction Result Box ── */
.pred-box {
    border-radius: 20px; padding: 2.2rem 2.5rem;
    margin: 1.2rem 0; position: relative; overflow: hidden;
}
.pred-box::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 100%; height: 100%; opacity: 0.04;
    background: repeating-linear-gradient(45deg, transparent,
    transparent 10px, currentColor 10px, currentColor 11px);
}
.pred-up {
    background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(16,185,129,0.04));
    border: 1px solid rgba(52,211,153,0.25);
}
.pred-dn {
    background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(239,68,68,0.04));
    border: 1px solid rgba(248,113,113,0.25);
}

/* ── Stock Card ── */
.stk-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 0.9rem 1.1rem;
    transition: all 0.3s ease;
}
.stk-card:hover {
    background: rgba(102,126,234,0.06);
    border-color: rgba(102,126,234,0.2);
}
.stk-sym { font-size: 0.95rem; font-weight: 800; color: #e2e8f0; }
.stk-name { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
.sector-tag {
    display: inline-block; font-size: 0.62rem; padding: 0.15rem 0.55rem;
    border-radius: 50px; background: rgba(102,126,234,0.1);
    color: #94a3b8; font-weight: 600; margin-top: 0.4rem;
}

/* ── Step Card ── */
.step-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.5rem; text-align: center;
    transition: all 0.3s ease; height: 100%;
}
.step-card:hover {
    border-color: rgba(102,126,234,0.25);
    transform: translateY(-2px);
}
.step-icon { font-size: 2.2rem; margin-bottom: 0.6rem; }
.step-title { font-weight: 700; color: #e2e8f0; font-size: 0.9rem; margin-bottom: 0.3rem; }
.step-desc { font-size: 0.78rem; color: #64748b; line-height: 1.5; }

/* ── Divider ── */
.divider {
    height: 1px; margin: 1.8rem 0;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.25), transparent);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(102,126,234,0.1);
}

/* ── Footer ── */
.footer {
    text-align: center; padding: 2.5rem 0 1rem 0;
    color: #475569; font-size: 0.78rem;
    border-top: 1px solid rgba(255,255,255,0.04); margin-top: 3rem;
}

/* ── About Table ── */
.info-table { width: 100%; color: #94a3b8; font-size: 0.85rem; }
.info-table td { padding: 0.4rem 0; }
.info-table .tv { text-align: right; color: #e2e8f0; font-weight: 700; }

/* ── Feature Tags ── */
.feat-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.feat-tag {
    display: inline-block; font-size: 0.68rem; padding: 0.2rem 0.6rem;
    border-radius: 50px; font-weight: 600;
    background: rgba(102,126,234,0.1); color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 0.5rem 0;">
        <div style="font-size:2.8rem;">📈</div>
        <h2 style="margin:0.4rem 0 0 0; font-size:1.35rem; font-weight:900;
                    background:linear-gradient(135deg,#667eea,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            StockVision AI
        </h2>
        <p style="color:#64748b; font-size:0.75rem; margin-top:0.2rem; letter-spacing:0.5px;">
            NIFTY50 Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", ["🏠 Dashboard", "🔮 Predict", "📊 Analytics", "ℹ️ About"],
                    label_visibility="collapsed")
    st.markdown("---")

    st.markdown("""
    <div style="padding:0.85rem; background:rgba(102,126,234,0.06);
                border-radius:12px; border:1px solid rgba(102,126,234,0.12);">
        <p style="font-size:0.68rem; color:#64748b; margin:0; line-height:1.6;">
            ⚠️ <strong style="color:#94a3b8;">Disclaimer</strong><br>
            Predictions are for educational purposes only.
            Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cached_load_data():
    return load_data()

@st.cache_data
def cached_get_stocks():
    return get_available_stocks()

@st.cache_data
def cached_features(_df, sym):
    return create_features(_df, sym)

df = cached_load_data()
available_stocks = cached_get_stocks()


# ═══════════════════════════════════════════════════════════════════════════════
# 🏠 DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🤖 Deep Learning Powered</div>
        <div class="hero-title">StockVision AI</div>
        <div class="hero-sub">
            Next-generation NIFTY50 stock prediction platform.<br>
            Powered by GRU neural networks trained on years of historical market data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="g-card">
            <div class="label">Trained Models</div>
            <div class="value val-blue">{len(available_stocks)}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="g-card">
            <div class="label">Architecture</div>
            <div class="value val-purple">GRU</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="g-card">
            <div class="label">Total Data Points</div>
            <div class="value val-green">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="g-card">
            <div class="label">Target</div>
            <div class="value" style="font-size:1.2rem;">Return-Based</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Stocks by Sector
    st.markdown('<div class="sec-title">📋 Available Stocks by Sector</div>', unsafe_allow_html=True)
    sectors = {}
    for s in available_stocks:
        sec = get_sector(s)
        sectors.setdefault(sec, []).append(s)

    for sec_name in sorted(sectors.keys()):
        with st.expander(f"**{sec_name}** — {len(sectors[sec_name])} stocks"):
            cols = st.columns(min(4, len(sectors[sec_name])))
            for i, stk in enumerate(sectors[sec_name]):
                with cols[i % 4]:
                    st.markdown(f"""<div class="stk-card">
                        <div class="stk-sym">{stk}</div>
                        <div class="stk-name">{get_display_name(stk)}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="sec-title">🧠 How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("📊", "Data Collection", "Historical OHLCV data from NSE for NIFTY50 stocks"),
        ("⚙️", "Feature Engineering", "MA, MACD, volatility, price ratios & more"),
        ("🤖", "GRU Training", "3-layer GRU neural network per stock"),
        ("🔮", "Prediction", "Next-day return → converted to price"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""<div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2.2rem; font-weight:900;
                    background:linear-gradient(135deg,#667eea,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    margin-bottom:0.2rem;">🔮 Stock Price Prediction</h1>
        <p style="color:#64748b; font-size:1rem;">
            Select a stock to get AI-powered next-day price predictions
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stock selector
    col_sel, col_inf = st.columns([2, 3])
    with col_sel:
        opts = {f"{s} — {get_display_name(s)}": s for s in available_stocks}
        sel_display = st.selectbox("Select Stock", list(opts.keys()), index=0)
        selected = opts[sel_display]

    with col_inf:
        st.markdown(f"""<div class="g-card" style="margin-top:0.5rem;">
            <span class="sector-tag">{get_sector(selected)}</span>
            <div style="font-size:1.3rem; font-weight:800; color:#e2e8f0; margin-top:0.3rem;">
                {get_display_name(selected)}
            </div>
            <div style="font-size:0.82rem; color:#64748b;">NSE: {selected}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Load data
    with st.spinner(f"Loading {selected}..."):
        data = cached_features(df, selected)
        summary = get_stock_summary(data)

    # ── Market Data Row ──
    st.markdown('<div class="sec-title">💰 Current Market Data</div>', unsafe_allow_html=True)

    pc = "val-green" if summary['change'] >= 0 else "val-red"
    arrow = "▲" if summary['change'] >= 0 else "▼"
    pill_cls = "pill-up" if summary['change'] >= 0 else "pill-dn"

    mc = st.columns(6)
    cards = [
        ("Close Price", f"₹{summary['close']:.2f}", pc,
         f'<div class="pill {pill_cls}">{arrow} {abs(summary["change_pct"]):.2f}%</div>'),
        ("Open", f"₹{summary['open']:.2f}", "", ""),
        ("Day High", f"₹{summary['high']:.2f}", "val-green", ""),
        ("Day Low", f"₹{summary['low']:.2f}", "val-red", ""),
        ("52W High", f"₹{summary['high_52w']:.2f}", "", ""),
        ("52W Low", f"₹{summary['low_52w']:.2f}", "", ""),
    ]
    for i, (lbl, val, cls, extra) in enumerate(cards):
        with mc[i]:
            sz = 'font-size:1.85rem;' if i == 0 else 'font-size:1.45rem;'
            st.markdown(f"""<div class="g-card">
                <div class="label">{lbl}</div>
                <div class="value {cls}" style="{sz}">{val}</div>
                {extra}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Historical Chart ──
    st.markdown('<div class="sec-title">📈 Historical Price Chart</div>', unsafe_allow_html=True)

    period = st.radio("Period", ["6M", "1Y", "3Y", "5Y", "All"], horizontal=True, index=1)
    pmap = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "All": len(data)}
    cdata = data.tail(min(pmap[period], len(data)))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=cdata['Date'], open=cdata['Open'], high=cdata['High'],
        low=cdata['Low'], close=cdata['Close'], name='OHLC',
        increasing_line_color='#34d399', decreasing_line_color='#f87171'))
    fig.add_trace(go.Scatter(x=cdata['Date'], y=cdata['MA10'],
        name='MA10', line=dict(color='#667eea', width=1.5)))
    fig.add_trace(go.Scatter(x=cdata['Date'], y=cdata['MA50'],
        name='MA50', line=dict(color='#a78bfa', width=1.5)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False,
        height=450, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter"))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Prediction ──
    st.markdown('<div class="sec-title">🔮 AI Prediction</div>', unsafe_allow_html=True)

    p1, p2 = st.columns([1, 1])
    with p1:
        if st.button("⚡ Predict Next Day", use_container_width=True, type="primary"):
            with st.spinner("Running GRU model..."):
                mdl, sx, sy = load_stock_model(selected)
                pred = predict_next_day(data, mdl, sx, sy)
                st.session_state['pred'] = pred
                st.session_state['pred_stock'] = selected
    with p2:
        ndays = st.slider("Multi-day forecast", 1, 7, 3)
        if st.button("📅 Predict Multiple Days", use_container_width=True):
            with st.spinner(f"Predicting {ndays} days..."):
                mdl, sx, sy = load_stock_model(selected)
                mpreds = predict_multiple_days(data, mdl, sx, sy, ndays)
                st.session_state['mpreds'] = mpreds
                st.session_state['mpreds_stock'] = selected

    # Single-day result
    if 'pred' in st.session_state and st.session_state.get('pred_stock') == selected:
        p = st.session_state['pred']
        dc = "#34d399" if p['direction'] == 'UP' else "#f87171"
        di = "📈" if p['direction'] == 'UP' else "📉"
        box_cls = "pred-up" if p['direction'] == 'UP' else "pred-dn"

        st.markdown(f"""
        <div class="pred-box {box_cls}">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; position:relative;">
                <div>
                    <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;
                                letter-spacing:2px; font-weight:700;">Predicted Next-Day Price</div>
                    <div style="font-size:3.2rem; font-weight:900; color:{dc}; margin:0.3rem 0;
                                letter-spacing:-1px;">₹{p['predicted_price']:.2f}</div>
                    <div style="font-size:0.95rem; color:#94a3b8;">
                        Current: ₹{p['current_price']:.2f}
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:4.5rem; line-height:1;">{di}</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{dc};">{p['change_pct']:+.2f}%</div>
                    <div style="font-size:0.95rem; color:{dc};">{p['change_amount']:+.2f} ₹</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Multi-day result
    if 'mpreds' in st.session_state and st.session_state.get('mpreds_stock') == selected:
        preds = st.session_state['mpreds']
        st.markdown(f'<div class="sec-title">📅 {len(preds)}-Day Forecast</div>', unsafe_allow_html=True)

        cols = st.columns(min(len(preds), 7))
        for i, pr in enumerate(preds):
            with cols[i % len(cols)]:
                cc = "#34d399" if pr['predicted_return'] >= 0 else "#f87171"
                ar = "▲" if pr['predicted_return'] >= 0 else "▼"
                st.markdown(f"""<div class="g-card" style="text-align:center;">
                    <div class="label">Day {pr['day']}</div>
                    <div style="font-size:0.72rem; color:#475569;">{pr['date'].strftime('%b %d')}</div>
                    <div style="font-size:1.45rem; font-weight:800; color:{cc}; margin:0.4rem 0;">
                        ₹{pr['predicted_price']:.2f}</div>
                    <div style="font-size:0.82rem; color:{cc};">{ar} {abs(pr['change_pct']):.2f}%</div>
                </div>""", unsafe_allow_html=True)

        fdf = pd.DataFrame(preds)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=fdf['date'], y=fdf['predicted_price'],
            mode='lines+markers', name='Predicted',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color='#667eea')))
        fig_f.add_hline(y=summary['current_price'], line_dash="dash",
            line_color="#475569",
            annotation_text=f"Current: ₹{summary['current_price']:.2f}")
        fig_f.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=350,
            margin=dict(l=0, r=0, t=30, b=0), font=dict(family="Inter"),
            yaxis_title="Price (₹)")
        fig_f.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
        fig_f.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Backtest ──
    st.markdown('<div class="sec-title">📊 Model Backtest</div>', unsafe_allow_html=True)
    if st.button("▶ Run Backtest", use_container_width=True):
        with st.spinner("Backtesting..."):
            mdl, sx, sy = load_stock_model(selected)
            bt = get_backtest_results(data, mdl, sx, sy)
            st.session_state['bt'] = bt
            st.session_state['bt_stock'] = selected

    if 'bt' in st.session_state and st.session_state.get('bt_stock') == selected:
        bt = st.session_state['bt']
        bm = st.columns(4)
        bt_items = [
            ("MAE", f"₹{bt['mae']:.2f}", "val-blue"),
            ("MAPE", f"{bt['mape']:.2f}%", "val-purple"),
            ("Direction Acc.", f"{bt['direction_accuracy']:.1f}%",
             "val-green" if bt['direction_accuracy'] >= 50 else "val-red"),
            ("Test Points", str(bt['test_size']), ""),
        ]
        for i, (lbl, val, cls) in enumerate(bt_items):
            with bm[i]:
                st.markdown(f"""<div class="g-card">
                    <div class="label">{lbl}</div>
                    <div class="value {cls}" style="font-size:1.4rem;">{val}</div>
                </div>""", unsafe_allow_html=True)

        cdf = bt['comparison_df']
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=cdf['Date'], y=cdf['Actual Price'],
            name='Actual', line=dict(color='#60a5fa', width=2)))
        fig_bt.add_trace(go.Scatter(x=cdf['Date'], y=cdf['Predicted Price'],
            name='Predicted', line=dict(color='#f87171', width=2, dash='dash')))
        fig_bt.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter"), yaxis_title="Price (₹)")
        fig_bt.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
        fig_bt.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig_bt, use_container_width=True)

    # Recent Data
    st.markdown('<div class="sec-title">📋 Recent Data</div>', unsafe_allow_html=True)
    show_cols = ['Date','Open','High','Low','Close','Volume','return','MA10','MA50','volatility']
    avail = [c for c in show_cols if c in data.columns]
    st.dataframe(data[avail].tail(10).style.format({
        'Open':'₹{:.2f}','High':'₹{:.2f}','Low':'₹{:.2f}',
        'Close':'₹{:.2f}','Volume':'{:,.0f}',
        'return':'{:.4f}','MA10':'₹{:.2f}','MA50':'₹{:.2f}','volatility':'{:.2f}'
    }), use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2.2rem; font-weight:900;
                    background:linear-gradient(135deg,#667eea,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            📊 Market Analytics</h1>
        <p style="color:#64748b;">Compare stocks, analyze sectors, and explore trends</p>
    </div>
    """, unsafe_allow_html=True)

    # Sector chart
    st.markdown('<div class="sec-title">🏢 Sector Distribution</div>', unsafe_allow_html=True)
    sec_counts = {}
    for s in available_stocks:
        sec_counts[get_sector(s)] = sec_counts.get(get_sector(s), 0) + 1
    sdf = pd.DataFrame({'Sector': list(sec_counts.keys()), 'Count': list(sec_counts.values())})
    sdf = sdf.sort_values('Count', ascending=True)

    fig_s = px.bar(sdf, x='Count', y='Sector', orientation='h',
        color='Count', color_continuous_scale=['#667eea', '#a78bfa'])
    fig_s.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=380,
        margin=dict(l=0, r=0, t=10, b=0), font=dict(family="Inter"),
        coloraxis_showscale=False)
    fig_s.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
    fig_s.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Comparison
    st.markdown('<div class="sec-title">📈 Stock Comparison</div>', unsafe_allow_html=True)
    cmp_stocks = st.multiselect("Select stocks to compare", available_stocks,
        default=available_stocks[:3], max_selections=6)

    if cmp_stocks:
        fig_c = go.Figure()
        palette = ['#667eea','#34d399','#f87171','#a78bfa','#fbbf24','#60a5fa']
        for i, stk in enumerate(cmp_stocks):
            sd = cached_features(df, stk)
            norm = (sd['Close'] / sd['Close'].iloc[0] - 1) * 100
            fig_c.add_trace(go.Scatter(x=sd['Date'], y=norm, name=stk,
                line=dict(color=palette[i % len(palette)], width=2)))
        fig_c.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=450,
            margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Cumulative Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter"))
        fig_c.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
        fig_c.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Volume
    if cmp_stocks:
        st.markdown('<div class="sec-title">📊 Avg Volume (Last 30 Days)</div>', unsafe_allow_html=True)
        vd = [{'Stock': s, 'Volume': cached_features(df, s)['Volume'].tail(30).mean()}
              for s in cmp_stocks]
        vdf = pd.DataFrame(vd).sort_values('Volume', ascending=True)
        fig_v = px.bar(vdf, x='Volume', y='Stock', orientation='h',
            color='Volume', color_continuous_scale=['#667eea','#a78bfa'])
        fig_v.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=280,
            margin=dict(l=0, r=0, t=10, b=0), font=dict(family="Inter"),
            coloraxis_showscale=False)
        fig_v.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
        fig_v.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig_v, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ℹ️ ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2.2rem; font-weight:900;
                    background:linear-gradient(135deg,#667eea,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            ℹ️ About This Project</h1>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""<div class="g-card">
            <div style="font-size:1.15rem; font-weight:800; color:#e2e8f0; margin-bottom:0.8rem;">
                🎯 Project Overview</div>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.88rem;">
                <strong>StockVision AI</strong> uses GRU (Gated Recurrent Unit) deep learning 
                to forecast next-day stock returns for NIFTY50 companies.</p>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.88rem; margin-top:0.5rem;">
                Instead of predicting raw prices, the model predicts <strong>returns</strong> 
                (percentage change) — which are stationary and scale-independent.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="g-card">
            <div style="font-size:1.15rem; font-weight:800; color:#e2e8f0; margin-bottom:0.8rem;">
                📊 Data Source</div>
            <p style="color:#94a3b8; line-height:1.7; font-size:0.88rem;">
                Historical data from <strong>NSE</strong> — OHLCV, VWAP, 
                and Deliverable Volume for all NIFTY50 stocks.</p>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""<div class="g-card">
            <div style="font-size:1.15rem; font-weight:800; color:#e2e8f0; margin-bottom:0.8rem;">
                🧠 Model Architecture</div>
            <table class="info-table">
                <tr><td>Model</td><td class="tv">GRU (3-Layer)</td></tr>
                <tr><td>Units</td><td class="tv">128 → 64 → 32</td></tr>
                <tr><td>Loss</td><td class="tv">MSE</td></tr>
                <tr><td>Optimizer</td><td class="tv">Adam</td></tr>
                <tr><td>Scaler</td><td class="tv">MinMaxScaler</td></tr>
                <tr><td>Lookback</td><td class="tv">60 days</td></tr>
                <tr><td>Target</td><td class="tv">Next-day Return</td></tr>
                <tr><td>Regularization</td><td class="tv">Dropout + BatchNorm</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="g-card">
            <div style="font-size:1.15rem; font-weight:800; color:#e2e8f0; margin-bottom:0.8rem;">
                ⚙️ Features (14)</div>
            <div class="feat-tags">
                <span class="feat-tag">Open</span><span class="feat-tag">High</span>
                <span class="feat-tag">Low</span><span class="feat-tag">Close</span>
                <span class="feat-tag">VWAP</span><span class="feat-tag">Volume</span>
                <span class="feat-tag">Del. Volume</span><span class="feat-tag">Return</span>
                <span class="feat-tag">MA10</span><span class="feat-tag">MA50</span>
                <span class="feat-tag">Volatility</span><span class="feat-tag">MACD</span>
                <span class="feat-tag">H/L Ratio</span><span class="feat-tag">C/O Ratio</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="g-card" style="margin-top:0.5rem;">
        <div style="font-size:1.15rem; font-weight:800; color:#e2e8f0; margin-bottom:0.5rem;">
            👨‍💻 Developed By</div>
        <p style="color:#e2e8f0; font-size:1rem; font-weight:600;">Nishant</p>
        <p style="color:#64748b; font-size:0.82rem; margin-top:0.2rem;">
            Built with Python • TensorFlow • Streamlit • Plotly</p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <p style="font-weight:600;">StockVision AI — NIFTY50 Intelligence Platform</p>
    <p style="margin-top:0.3rem;">Built with ❤️ using Python • TensorFlow • Streamlit • Plotly</p>
</div>
""", unsafe_allow_html=True)
