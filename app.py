"""NeuroPulse AI - Streamlit real-time neural signal dashboard."""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ml_pipeline import NeuroPulsePipeline, CHANNELS, FS, BANDS, STATES

st.set_page_config(
    page_title="NeuroPulse AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Palette --------------------------------------------------------------- #
NEON = ["#22d3ee", "#818cf8", "#34d399", "#fbbf24", "#fb7185", "#2dd4bf"]
BAND_COLORS = ["#22d3ee", "#818cf8", "#34d399", "#fbbf24", "#fb7185"]
RISK_COLORS = {"Low": "#34d399", "Medium": "#fbbf24", "Critical": "#fb7185"}
ELECTRODES = {
    "Fp1": (-0.45, 0.82), "Fp2": (0.45, 0.82),
    "C3": (-0.72, 0.02), "C4": (0.72, 0.02),
    "P3": (-0.42, -0.72), "P4": (0.42, -0.72),
}

# --------------------------------------------------------------------------- #
# Custom dark "neuro" theme
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #070b14;
        --panel: rgba(17, 24, 42, 0.66);
        --panel-strong: rgba(22, 30, 52, 0.88);
        --stroke: rgba(99, 140, 255, 0.22);
        --text: #e8eefc;
        --muted: #8ea0c4;
        --cyan: #22d3ee;
        --violet: #818cf8;
        --green: #34d399;
        --amber: #fbbf24;
        --rose: #fb7185;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(900px 520px at 8% -8%, rgba(34, 211, 238, 0.16), transparent 60%),
            radial-gradient(880px 560px at 96% 0%, rgba(129, 140, 248, 0.18), transparent 62%),
            radial-gradient(760px 620px at 50% 108%, rgba(251, 113, 133, 0.12), transparent 60%),
            #070b14;
        background-size: 180% 180%;
        animation: aurora 22s ease-in-out infinite;
    }
    @keyframes aurora {
        0%, 100% { background-position: 0% 0%, 100% 0%, 50% 100%; }
        50% { background-position: 40% 20%, 60% 30%, 40% 70%; }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12,18,33,0.98), rgba(10,15,28,0.98));
        border-right: 1px solid var(--stroke);
        width: 310px !important; min-width: 310px !important;
    }
    section[data-testid="stSidebar"] > div { width: 310px !important; }
    .block-container { padding-top: 1.1rem; max-width: 1500px; }

    /* Hero */
    .hero {
        display: flex; align-items: center; justify-content: space-between;
        gap: 18px; flex-wrap: wrap;
        padding: 20px 24px; margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(17,24,42,0.82), rgba(17,24,42,0.38));
        border: 1px solid var(--stroke); border-radius: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 18px 50px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-left { display: flex; align-items: center; gap: 16px; }
    .logo-orb {
        width: 58px; height: 58px; border-radius: 18px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        background: radial-gradient(circle at 30% 25%, rgba(34,211,238,0.45), rgba(129,140,248,0.25));
        border: 1px solid rgba(34,211,238,0.45);
        box-shadow: 0 0 28px rgba(34,211,238,0.35);
        animation: orb 5s ease-in-out infinite;
    }
    @keyframes orb {
        0%, 100% { transform: translateY(0); box-shadow: 0 0 24px rgba(34,211,238,0.30); }
        50% { transform: translateY(-3px); box-shadow: 0 0 38px rgba(129,140,248,0.45); }
    }
    .np-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.15rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1;
        background: linear-gradient(90deg, #67e8f9, #a5b4fc, #f0abfc, #67e8f9);
        background-size: 260% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: shimmer 7s linear infinite;
    }
    @keyframes shimmer { to { background-position: 260% center; } }
    .np-sub { color: var(--muted); font-size: 0.9rem; margin-top: 8px; }
    .hero-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .live-pill {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 14px; border-radius: 999px;
        background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52,211,153,0.4);
        color: #6ee7b7; font-weight: 700; font-size: 0.78rem; letter-spacing: 1.4px;
    }
    .live-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #34d399;
        box-shadow: 0 0 0 0 rgba(52,211,153,0.7); animation: pulse-dot 1.6s infinite;
    }
    @keyframes pulse-dot {
        0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.65); }
        70% { box-shadow: 0 0 0 10px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }
    .chip {
        padding: 8px 12px; border-radius: 999px; font-size: 0.72rem;
        color: var(--muted); border: 1px solid var(--stroke);
        background: rgba(255,255,255,0.03);
    }

    /* KPI cards */
    .kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 4px; }
    .kpi-card {
        position: relative; overflow: hidden;
        flex: 1 1 220px; min-width: 210px;
        background: linear-gradient(150deg, rgba(24,33,56,0.82), rgba(13,19,34,0.72));
        border: 1px solid var(--stroke); border-radius: 18px;
        padding: 18px 18px 16px;
        backdrop-filter: blur(14px);
        box-shadow: 0 14px 34px rgba(0,0,0,0.35);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .kpi-card::before {
        content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
        background: linear-gradient(90deg, var(--cyan), var(--violet), var(--rose));
        opacity: 0.8;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(34,211,238,0.45);
        box-shadow: 0 20px 44px rgba(34,211,238,0.12);
    }
    .kpi-head { display: flex; align-items: center; justify-content: space-between; }
    .kpi-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.4px; }
    .kpi-icon { font-size: 1.05rem; opacity: 0.9; }
    .kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; margin-top: 8px; }
    .kpi-sub { color: var(--muted); font-size: 0.76rem; margin-top: 5px; }
    .risk-Low { color: var(--green); }
    .risk-Medium { color: var(--amber); }
    .risk-Critical { color: var(--rose); }
    .glow-Low { text-shadow: 0 0 18px rgba(52,211,153,0.5); }
    .glow-Medium { text-shadow: 0 0 18px rgba(251,191,36,0.5); }
    .glow-Critical { text-shadow: 0 0 18px rgba(251,113,133,0.65); }

    /* Section headings */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem; font-weight: 600; letter-spacing: 0.4px;
        color: #c7d5f5; margin: 6px 0 8px;
        display: flex; align-items: center; gap: 8px;
    }
    .section-title::before {
        content: ""; width: 3px; height: 15px; border-radius: 3px;
        background: linear-gradient(180deg, var(--cyan), var(--violet));
    }

    /* Alert feed */
    .alert-feed { max-height: 300px; overflow-y: auto; padding-right: 4px; }
    .alert-feed::-webkit-scrollbar { width: 6px; }
    .alert-feed::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.4); border-radius: 6px; }
    .alert-item {
        background: rgba(22, 30, 52, 0.8);
        border: 1px solid rgba(251,113,133,0.28);
        border-left: 3px solid var(--rose);
        padding: 9px 12px; margin-bottom: 8px; border-radius: 10px;
        font-size: 0.82rem; color: #dbe6ff;
        animation: slide-in .25s ease;
    }
    .alert-item.medium { border-left-color: var(--amber); border-color: rgba(251,191,36,0.25); }
    .alert-item.low { border-left-color: var(--green); border-color: rgba(52,211,153,0.22); }
    .alert-item.critical { animation: critical-pulse 1.4s ease-in-out infinite; }
    @keyframes critical-pulse {
        0%, 100% { box-shadow: 0 0 0 rgba(251,113,133,0); }
        50% { box-shadow: 0 0 18px rgba(251,113,133,0.28); }
    }
    @keyframes slide-in { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
    .alert-empty {
        background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.25);
        border-left: 3px solid var(--green); padding: 10px 12px; border-radius: 10px;
        font-size: 0.82rem; color: #a7f3d0;
    }

    /* Sidebar polish */
    .sidebar-stat {
        background: rgba(255,255,255,0.03); border: 1px solid var(--stroke);
        border-radius: 12px; padding: 10px 12px; margin: 6px 0; font-size: 0.8rem;
    }
    .sidebar-stat b { color: var(--cyan); }

    .stPlotlyChart, .element-container, div[data-testid="stVerticalBlock"] { transition: none !important; }
    [data-testid="stStatusWidget"], header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    div[data-testid="stMetric"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "pipeline" not in st.session_state:
    with st.spinner("Training NeuroPulse models on synthetic baseline sessions..."):
        st.session_state.pipeline = NeuroPulsePipeline()
if "t_offset" not in st.session_state:
    st.session_state.t_offset = 0.0
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "inject_now" not in st.session_state:
    st.session_state.inject_now = False
if "started_at" not in st.session_state:
    st.session_state.started_at = time.time()

pipeline = st.session_state.pipeline

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙️ Mission Control")
    st.caption("Tune the synthetic EEG stream and live ML inference.")
    noise = st.slider("Signal Noise", 0.0, 2.0, 0.5, 0.1, key="noise_slider")
    speed = st.slider("Simulation Speed (s/frame)", 0.2, 2.0, 0.6, 0.1, key="speed_slider")
    refresh = st.slider("Refresh Interval (s)", 1.0, 3.0, 2.0, 0.5, key="refresh_slider")
    inject_toggle = st.toggle("Anomaly Injection (continuous)", value=False, key="inject_toggle")
    if st.button("⚡ Inject Spike Now", use_container_width=True):
        st.session_state.inject_now = True

    st.markdown("---")
    st.markdown("### 🧠 State Control")
    auto_cycle = st.toggle("Auto-cycle States", value=False, key="auto_cycle_toggle")
    live_mode = st.toggle("Live Auto-Refresh", value=True, key="live_mode_toggle")
    state_select = st.selectbox("Simulated Brain State", STATES, index=0, key="state_select")

    st.markdown("---")
    st.markdown("### 📥 Reports")
    if st.button("Clear Logs", use_container_width=True):
        pipeline.clear_logs()
        st.session_state.t_offset = 0.0
        st.session_state.frame = 0
    df_logs = pd.DataFrame(pipeline.session_log)
    st.download_button(
        "⬇ Download Session CSV",
        data=df_logs.to_csv(index=False).encode("utf-8") if not df_logs.empty else b"",
        file_name="neuropulse_session_log.csv",
        mime="text/csv",
        disabled=df_logs.empty,
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown(
        f'<div class="sidebar-stat">Session frames: <b>{len(pipeline.session_log)}</b></div>'
        f'<div class="sidebar-stat">Anomalies logged: <b>{len(pipeline.anomaly_log)}</b></div>'
        f'<div class="sidebar-stat">Sample rate: <b>{FS} Hz</b></div>',
        unsafe_allow_html=True,
    )
    st.caption("NeuroPulse AI v2.0 · Synthetic EEG research demo")

# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <div class="hero-left">
        <div class="logo-orb">🧠</div>
        <div>
          <div class="np-title">NeuroPulse AI</div>
          <div class="np-sub">Real-Time Neural Signal Classifier &amp; Anomaly Detector ·
          6-channel synthetic EEG · IsolationForest + RandomForest</div>
        </div>
      </div>
      <div class="hero-right">
        <div class="live-pill"><span class="live-dot"></span> LIVE INFERENCE</div>
        <div class="chip">6 channels</div>
        <div class="chip">256 Hz</div>
        <div class="chip">0.5–45 Hz</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Persistent placeholders
# --------------------------------------------------------------------------- #
kpi_ph = st.empty()
row1_left, row1_right = st.columns([1.75, 1])
with row1_left:
    wave_ph = st.empty()
with row1_right:
    gauge_ph = st.empty()
    prob_ph = st.empty()

row2_a, row2_b, row2_c = st.columns(3)
with row2_a:
    band_ph = st.empty()
with row2_b:
    scalp_ph = st.empty()
with row2_c:
    heat_ph = st.empty()

row3_left, row3_right = st.columns([1.75, 1])
with row3_left:
    timeline_ph = st.empty()
with row3_right:
    feed_ph = st.empty()

# --------------------------------------------------------------------------- #
# Live fragment
# --------------------------------------------------------------------------- #
live_mode = st.session_state.get("live_mode_toggle", True)
refresh_val = st.session_state.get("refresh_slider", 2.0) if live_mode else None


def _base_layout(title, height):
    return dict(
        title=dict(text=title, font=dict(size=14, color="#c7d5f5")),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe6ff", family="Inter"),
        margin=dict(l=36, r=18, t=46, b=32),
        transition=dict(duration=0),
    )


@st.fragment(run_every=refresh_val)
def live_panel():
    pipeline = st.session_state.pipeline
    noise = st.session_state.get("noise_slider", 0.5)
    inject_toggle = st.session_state.get("inject_toggle", False)
    auto_cycle = st.session_state.get("auto_cycle_toggle", False)
    state_select = st.session_state.get("state_select", STATES[0])

    inject_now = st.session_state.inject_now
    st.session_state.inject_now = False
    do_inject = inject_toggle or inject_now

    if auto_cycle:
        current_state = STATES[st.session_state.frame % len(STATES)]
    else:
        current_state = state_select

    sig, t = pipeline.generate_signal(
        current_state, duration=2.0, noise=noise, inject_anomaly=do_inject,
        t_offset=st.session_state.t_offset,
    )
    res = pipeline.predict(sig, noise_level=noise)

    state = res["state"]
    risk = res["risk"]
    risk_pct = res["risk_pct"]
    dominant = res["dominant_band"]
    latency = res["latency_ms"]
    bp = res["band_power"]
    per_channel = res.get("channel_band_power", [])
    rf_proba = res["rf_proba"]

    icons = {
        "Current Brain State": "🧠",
        "Anomaly Risk": "🛡️",
        "Dominant Frequency Band": "📡",
        "Processing Latency": "⚡",
    }
    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-head"><div class="kpi-label">{list(icons)[0]}</div><div class="kpi-icon">{icons["Current Brain State"]}</div></div>
        <div class="kpi-value">{state}</div>
        <div class="kpi-sub">RandomForest · confidence {max(rf_proba.values())*100:.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-head"><div class="kpi-label">{list(icons)[1]}</div><div class="kpi-icon">{icons["Anomaly Risk"]}</div></div>
        <div class="kpi-value risk-{risk} glow-{risk}">{risk}</div>
        <div class="kpi-sub">{risk_pct:.1f}% · IsolationForest</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-head"><div class="kpi-label">{list(icons)[2]}</div><div class="kpi-icon">{icons["Dominant Frequency Band"]}</div></div>
        <div class="kpi-value">{dominant}</div>
        <div class="kpi-sub">{bp[dominant]:.2f} aggregate power</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-head"><div class="kpi-label">{list(icons)[3]}</div><div class="kpi-icon">{icons["Processing Latency"]}</div></div>
        <div class="kpi-value">{latency:.1f} ms</div>
        <div class="kpi-sub">Feature extraction + inference</div>
      </div>
    </div>
    """
    kpi_ph.markdown(kpi_html, unsafe_allow_html=True)

    # ---- waveform ------------------------------------------------------- #
    fig_wave = go.Figure()
    for ci, ch in enumerate(CHANNELS):
        fig_wave.add_trace(go.Scatter(
            x=t, y=sig[ci] + ci * 4.2, mode="lines", name=ch,
            line=dict(color=NEON[ci], width=1.3),
            hovertemplate=f"{ch}<br>%{{y:.2f}}<extra></extra>",
        ))
    fig_wave.update_layout(**_base_layout("⚡ Multi-Channel EEG Waveform", 420))
    fig_wave.update_xaxes(title="Time (s)", gridcolor="rgba(99,140,255,0.12)", zeroline=False)
    fig_wave.update_yaxes(title="Amplitude (offset)", gridcolor="rgba(99,140,255,0.10)", showticklabels=False)
    fig_wave.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=36, r=18, t=46, b=64),
    )
    wave_ph.plotly_chart(fig_wave, width="stretch", key="wave_chart",
                         config={"displayModeBar": False}, theme=None)

    # ---- risk gauge ----------------------------------------------------- #
    gauge_color = RISK_COLORS[risk]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 30, "color": gauge_color}},
        title={"text": "Anomaly Risk", "font": {"size": 13, "color": "#8ea0c4"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2a3a5c",
                     "tickfont": {"size": 9, "color": "#8ea0c4"}},
            "bar": {"color": gauge_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(52,211,153,0.16)"},
                {"range": [30, 70], "color": "rgba(251,191,36,0.14)"},
                {"range": [70, 100], "color": "rgba(251,113,133,0.16)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.75,
                          "value": risk_pct},
        },
    ))
    fig_gauge.update_layout(**_base_layout("", 200))
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=36, b=8))
    gauge_ph.plotly_chart(fig_gauge, width="stretch", key="gauge_chart",
                          config={"displayModeBar": False}, theme=None)

    # ---- state probabilities ------------------------------------------- #
    short = {"Calm / Relaxed": "Calm", "Focused / Active": "Focused",
             "Fatigued": "Fatigued", "Abnormal Spike": "Spike"}
    prob_labels = [short[s] for s in rf_proba.keys()]
    prob_keys = list(rf_proba.keys())
    prob_values = [rf_proba[s] * 100 for s in prob_keys]
    fig_prob = go.Figure(go.Bar(
        x=prob_values, y=prob_labels, orientation="h",
        marker=dict(color=[RISK_COLORS["Critical"] if s == "Abnormal Spike"
                           else NEON[i % len(NEON)] for i, s in enumerate(prob_keys)],
                    line=dict(width=0)),
        text=[f"{v:.0f}%" for v in prob_values], textposition="outside",
        textfont=dict(size=10, color="#dbe6ff"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_prob.update_layout(**_base_layout("🧠 Brain-State Probability", 210))
    fig_prob.update_xaxes(range=[0, 110], gridcolor="rgba(99,140,255,0.10)",
                          showticklabels=False, title=None)
    fig_prob.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10))
    fig_prob.update_layout(margin=dict(l=58, r=26, t=42, b=10))
    prob_ph.plotly_chart(fig_prob, width="stretch", key="prob_chart",
                         config={"displayModeBar": False}, theme=None)

    # ---- band power ----------------------------------------------------- #
    band_names = list(BANDS.keys())
    band_vals = [float(bp[b]) for b in band_names]
    fig_band = go.Figure(go.Bar(
        x=band_names, y=band_vals,
        marker=dict(color=BAND_COLORS, line=dict(width=0)),
        text=[f"{v:.2f}" for v in band_vals], textposition="outside",
        textfont=dict(size=10, color="#dbe6ff"),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig_band.update_layout(**_base_layout("📊 Band Power", 300))
    fig_band.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig_band.update_yaxes(title="Power", gridcolor="rgba(99,140,255,0.12)")
    band_ph.plotly_chart(fig_band, width="stretch", key="band_chart",
                         config={"displayModeBar": False}, theme=None)

    # ---- scalp topography ---------------------------------------------- #
    fig_scalp = go.Figure()
    # head outline
    theta = np.linspace(0, 2 * np.pi, 200)
    fig_scalp.add_trace(go.Scatter(
        x=np.cos(theta) * 1.02, y=np.sin(theta) * 1.02, mode="lines",
        line=dict(color="rgba(99,140,255,0.45)", width=2), hoverinfo="skip",
        showlegend=False,
    ))
    # nose
    fig_scalp.add_trace(go.Scatter(
        x=[-0.12, 0, 0.12], y=[0.99, 1.16, 0.99], mode="lines",
        line=dict(color="rgba(99,140,255,0.45)", width=2), hoverinfo="skip",
        showlegend=False,
    ))
    # ears
    for sign in (-1, 1):
        fig_scalp.add_trace(go.Scatter(
            x=[sign * 1.02, sign * 1.13, sign * 1.02],
            y=[0.26, 0, -0.26], mode="lines",
            line=dict(color="rgba(99,140,255,0.45)", width=2), hoverinfo="skip",
            showlegend=False,
        ))
    alpha_idx = band_names.index("Alpha")
    for ci, ch in enumerate(CHANNELS):
        if per_channel:
            val = per_channel[ci]["Alpha"]
        else:
            val = bp["Alpha"] / len(CHANNELS)
        x, y = ELECTRODES[ch]
        fig_scalp.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=20 + min(26, val * 1.6), color=NEON[ci],
                        line=dict(color="rgba(255,255,255,0.55)", width=1),
                        opacity=0.9),
            text=[ch], textposition="middle center",
            textfont=dict(size=9, color="#06101f", family="Inter"),
            name=ch, hovertemplate=f"{ch}<br>Alpha: {val:.2f}<extra></extra>",
            showlegend=False,
        ))
    fig_scalp.update_layout(**_base_layout("🧭 Scalp Map · Alpha", 300))
    fig_scalp.update_xaxes(range=[-1.35, 1.35], visible=False)
    fig_scalp.update_yaxes(range=[-1.28, 1.30], visible=False, scaleanchor="x", scaleratio=1)
    fig_scalp.update_layout(showlegend=False, margin=dict(l=8, r=8, t=42, b=8))
    scalp_ph.plotly_chart(fig_scalp, width="stretch", key="scalp_chart",
                          config={"displayModeBar": False}, theme=None)

    # ---- channel x band heatmap ---------------------------------------- #
    heat_z = []
    for ci in range(len(CHANNELS)):
        if per_channel:
            heat_z.append([per_channel[ci][b] for b in band_names])
        else:
            heat_z.append([bp[b] / len(CHANNELS) for b in band_names])
    fig_heat = go.Figure(go.Heatmap(
        z=heat_z, x=band_names, y=CHANNELS,
        colorscale=[[0, "#0b1020"], [0.35, "#1e3a8a"], [0.65, "#22d3ee"], [1, "#f0abfc"]],
        text=[[f"{v:.2f}" for v in row] for row in heat_z],
        texttemplate="%{text}", textfont=dict(size=9, color="#e8eefc"),
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=8, len=0.75, tickfont=dict(size=8, color="#8ea0c4")),
    ))
    fig_heat.update_layout(**_base_layout("🔥 Channel Heatmap", 300))
    fig_heat.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig_heat.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
    heat_ph.plotly_chart(fig_heat, width="stretch", key="heat_chart",
                         config={"displayModeBar": False}, theme=None)

    # ---- timeline ------------------------------------------------------- #
    if pipeline.session_log:
        sdf = pd.DataFrame(pipeline.session_log)
        sdf["idx"] = range(len(sdf))
        fig_tl = go.Figure()
        for risk_name in ["Low", "Medium", "Critical"]:
            sub = sdf[sdf["risk"] == risk_name]
            if not sub.empty:
                fig_tl.add_trace(go.Scatter(
                    x=sub["idx"], y=[risk_name] * len(sub), mode="markers",
                    name=risk_name,
                    marker=dict(color=RISK_COLORS[risk_name], size=9,
                                line=dict(color="rgba(255,255,255,0.35)", width=1)),
                    text=sub["state"],
                    hovertemplate="%{text}<br>risk=%{y}<extra></extra>",
                ))
        fig_tl.update_layout(**_base_layout("🕒 Anomaly Event Timeline", 300))
        fig_tl.update_xaxes(title="Frame", gridcolor="rgba(99,140,255,0.10)")
        fig_tl.update_yaxes(gridcolor="rgba(99,140,255,0.10)")
        fig_tl.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         xanchor="right", x=1, font=dict(size=10)))
        timeline_ph.plotly_chart(fig_tl, width="stretch", key="timeline_chart",
                                 config={"displayModeBar": False}, theme=None)

    # ---- alert feed ----------------------------------------------------- #
    recent = list(reversed(pipeline.anomaly_log[-10:]))
    if recent:
        items = []
        for ev in recent:
            cls = "alert-item"
            if ev["risk"] == "Medium":
                cls += " medium"
            elif ev["risk"] == "Low":
                cls += " low"
            elif ev["risk"] == "Critical":
                cls += " critical"
            items.append(
                f'<div class="{cls}"><b>{ev["timestamp"]}</b><br>'
                f'{ev["state"]} · <span class="risk-{ev["risk"]}">{ev["risk"]}</span> '
                f'({ev["risk_pct"]}%) · {ev["dominant_band"]}</div>'
            )
        feed_html = '<div class="alert-feed">' + "".join(items) + "</div>"
    else:
        feed_html = ('<div class="alert-feed"><div class="alert-empty">'
                     'All clear — no anomalies detected yet.</div></div>')
    feed_ph.markdown(
        f'<div class="section-title">🚨 Live Anomaly Feed</div>{feed_html}',
        unsafe_allow_html=True,
    )

    st.session_state.t_offset += 2.0
    st.session_state.frame += 1


live_panel()