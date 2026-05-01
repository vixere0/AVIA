"""
AVIA - Autonomous Vehicle Intelligence Agent
Run with: streamlit run avia_agent.py
Requirements: pip install torch streamlit langchain-groq langgraph langchain-core joblib scikit-learn plotly pandas python-dotenv shap
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

load_dotenv()


class DiagnosticNet(nn.Module):
    def __init__(self):
        super(DiagnosticNet, self).__init__()
        self.layer1   = nn.Linear(5, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2   = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3   = nn.Linear(16, 8)
        self.output   = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        return self.output(x)


@st.cache_resource
def load_model_and_scaler():
    model = DiagnosticNet()
    model.load_state_dict(torch.load('diagnostic_model.pth', map_location='cpu'))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    return model, scaler


try:
    nn_model, scaler = load_model_and_scaler()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing_file = str(e)


SENSOR_RANGES = {
    "Air Temp (K)":       {"min": 295.0, "max": 305.0, "unit": "K"},
    "Process Temp (K)":   {"min": 305.0, "max": 315.0, "unit": "K"},
    "Engine Speed (RPM)": {"min": 1200,  "max": 2500,  "unit": "RPM"},
    "Torque (Nm)":        {"min": 20.0,  "max": 60.0,  "unit": "Nm"},
    "Tool Wear (min)":    {"min": 0,     "max": 200,   "unit": "min"},
}


@tool
def check_vehicle_health(
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    wear: float
) -> dict:
    """
    Analyzes vehicle sensor data using a PyTorch model and returns the failure probability.

    Parameters:
      air_temp     : Air temperature (Kelvin), e.g. 298.1
      process_temp : Process temperature (Kelvin), e.g. 308.6
      rpm          : Engine speed (RPM), e.g. 1551
      torque       : Torque (Nm), e.g. 42.8
      wear         : Tool wear (minutes), e.g. 0
    """
    if not model_loaded:
        return {"error": "Model could not be loaded. Run train_model.py first."}

    raw = np.array([[air_temp, process_temp, rpm, torque, wear]], dtype=np.float32)
    scaled = scaler.transform(raw)
    inputs = torch.FloatTensor(scaled)

    with torch.no_grad():
        logit = nn_model(inputs)
        probability = torch.sigmoid(logit).item()

    status = "CRITICAL" if probability > 0.5 else "STABLE"

    return {
        "failure_probability_pct": round(probability * 100, 2),
        "status": status,
        "raw_probability": round(probability, 4),
        "input_summary": {
            "air_temp_K": air_temp,
            "process_temp_K": process_temp,
            "engine_rpm": rpm,
            "torque_Nm": torque,
            "wear_min": wear
        }
    }


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_KEY")


@st.cache_resource
def create_agent():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    system_prompt = """You are an experienced industrial maintenance engineer.
Your job: analyze sensor data using the 'check_vehicle_health' tool.

RULES:
1. ALWAYS call 'check_vehicle_health' for every analysis.
2. Based on the 'failure_probability_pct' returned by the tool:
   - ABOVE 50% → Header: "🔴 CRITICAL CONDITION"
   - BELOW 50% → Header: "🟢 SYSTEM STABLE"
3. Never interpret results in a way that contradicts the tool output.
4. Use these section headers:
   - **Failure Probability**: (percentage value)
   - **Assessment**: (brief status description)
   - **Recommended Action**: (what should be done)
   - **Risk Factors**: (which sensors appear abnormal)
5. Respond in English."""

    agent = create_react_agent(
        llm,
        tools=[check_vehicle_health],
        prompt=system_prompt
    )
    return agent


def make_gauge(probability_pct: float) -> go.Figure:
    color = "#ff4444" if probability_pct > 50 else "#00c851"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability_pct,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={
            "reference": 50,
            "increasing": {"color": "#ff4444"},
            "decreasing": {"color": "#00c851"}
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "#0d2e1a"},
                {"range": [30, 50],  "color": "#2e2a0d"},
                {"range": [50, 75],  "color": "#2e1a0d"},
                {"range": [75, 100], "color": "#2e0d0d"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": 50
            }
        },
        title={"text": "Failure Probability", "font": {"size": 14, "color": "#888"}}
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20, l=30, r=30),
        height=250,
        font={"color": "#ccc"}
    )
    return fig


def make_sensor_radar(values: dict) -> go.Figure:
    sensors  = list(SENSOR_RANGES.keys())
    raw_vals = [
        values["air_temp_K"],
        values["process_temp_K"],
        values["engine_rpm"],
        values["torque_Nm"],
        values["wear_min"],
    ]

    def normalize(val, rng):
        mid  = (rng["min"] + rng["max"]) / 2
        span = rng["max"] - rng["min"]
        return max(0, min(100, 50 + (val - mid) / span * 100))

    norm_vals  = [normalize(v, SENSOR_RANGES[s]) for v, s in zip(raw_vals, sensors)]
    norm_ideal = [50] * len(sensors)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_ideal + [norm_ideal[0]],
        theta=sensors + [sensors[0]],
        fill='toself',
        fillcolor='rgba(0,200,81,0.08)',
        line=dict(color='#00c851', width=1, dash='dash'),
        name='Normal Range'
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=sensors + [sensors[0]],
        fill='toself',
        fillcolor='rgba(0,212,255,0.12)',
        line=dict(color='#00d4ff', width=2),
        name='Current Readings'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color="#666"), gridcolor="#2a2a2a"),
            angularaxis=dict(tickfont=dict(size=10, color="#aaa"), gridcolor="#2a2a2a"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#aaa"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=40, r=40),
        height=280,
        font={"color": "#ccc"}
    )
    return fig


def make_sensor_bars(values: dict) -> go.Figure:
    sensors  = list(SENSOR_RANGES.keys())
    raw_vals = [
        values["air_temp_K"],
        values["process_temp_K"],
        values["engine_rpm"],
        values["torque_Nm"],
        values["wear_min"],
    ]

    colors, statuses = [], []
    for val, sensor in zip(raw_vals, sensors):
        rng = SENSOR_RANGES[sensor]
        if rng["min"] <= val <= rng["max"]:
            colors.append("#00c851"); statuses.append("Normal")
        elif val < rng["min"]:
            colors.append("#ffbb33"); statuses.append("Low")
        else:
            colors.append("#ff4444"); statuses.append("High")

    def normalize(val, rng):
        span = rng["max"] - rng["min"]
        return max(0, min(100, (val - rng["min"]) / span * 100)) if span > 0 else 50

    norm_vals = [normalize(v, SENSOR_RANGES[s]) for v, s in zip(raw_vals, sensors)]

    fig = go.Figure(go.Bar(
        x=norm_vals,
        y=sensors,
        orientation='h',
        marker_color=colors,
        text=[f"{v} {SENSOR_RANGES[s]['unit']}  [{st}]"
              for v, s, st in zip(raw_vals, sensors, statuses)],
        textposition='inside',
        textfont=dict(size=11, color="#fff"),
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.add_vrect(x0=0, x1=100, fillcolor="rgba(0,200,81,0.04)", line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 120], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#aaa", size=11), gridcolor="#2a2a2a"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        showlegend=False,
        bargap=0.35,
    )
    return fig


def make_history_chart(history: list) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(255,68,68,0.05)", line_width=0)
    fig.add_hline(y=50, line_color="#555", line_dash="dash", line_width=1,
                  annotation_text="Risk Threshold", annotation_font_color="#666")
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["probability"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(
            color=["#ff4444" if p > 50 else "#00c851" for p in df["probability"]],
            size=9,
            line=dict(color="#111", width=1)
        ),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.06)',
        hovertemplate='%{x}<br>Probability: %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#666", size=10), gridcolor="#1e1e1e"),
        yaxis=dict(range=[0, 105], tickfont=dict(color="#666", size=10),
                   gridcolor="#1e1e1e", ticksuffix="%"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        showlegend=False,
    )
    return fig


SENSOR_LABELS = [
    "Air Temp (K)",
    "Process Temp (K)",
    "Engine Speed (RPM)",
    "Torque (Nm)",
    "Tool Wear (min)",
]


def compute_shap_values(raw_input: np.ndarray) -> np.ndarray | None:
    """
    DiagnosticNet için KernelExplainer ile SHAP değerlerini hesaplar.
    TensorFlow veya GPU bağımlılığı yoktur — sadece numpy/torch yeterli.
    raw_input : (1, 5) ölçeksiz numpy array
    Döndürür  : (5,) array — her sensörün SHAP katkısı (sigmoid çıktısına göre)
    """
    if not shap_available or not model_loaded:
        return None

    scaled = scaler.transform(raw_input).astype(np.float32)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t   = torch.FloatTensor(x.astype(np.float32))
            out = torch.sigmoid(nn_model(t)).numpy().flatten()
        return out

    background = np.zeros((1, 5), dtype=np.float32)
    explainer  = shap.KernelExplainer(predict_fn, background)
    shap_vals  = explainer.shap_values(scaled, nsamples=128, silent=True)
    return np.array(shap_vals).flatten()


def make_shap_chart(shap_values: np.ndarray, prob_pct: float) -> go.Figure:
    """
    SHAP değerlerini yatay waterfall benzeri bar chart olarak gösterir.
    Pozitif değer → arıza olasılığını artırıyor (kırmızı)
    Negatif değer → arıza olasılığını düşürüyor (yeşil)
    Barlar mutlak değere göre küçükten büyüğe sıralanır.
    """
    sorted_pairs = sorted(zip(shap_values.tolist(), SENSOR_LABELS), key=lambda x: abs(x[0]))
    s_vals, s_labels = zip(*sorted_pairs)

    colors = ["#ff4444" if v > 0 else "#00c851" for v in s_vals]
    texts  = [f"+{v:.4f}" if v > 0 else f"{v:.4f}" for v in s_vals]

    fig = go.Figure()
    fig.add_vline(x=0, line_color="#333", line_width=1)
    fig.add_trace(go.Bar(
        x=list(s_vals),
        y=list(s_labels),
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=texts,
        textposition="outside",
        textfont=dict(size=11, color="#aaa"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>",
        width=0.55,
    ))

    max_abs    = max(abs(v) for v in s_vals) if s_vals else 0.01
    axis_range = [-max_abs * 1.6, max_abs * 1.6]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=axis_range,
            showgrid=True, gridcolor="#1a1a1a", gridwidth=1, zeroline=False,
            tickfont=dict(color="#555", size=10),
            title=dict(text="← reduces risk  |  increases risk →",
                       font=dict(size=10, color="#444")),
        ),
        yaxis=dict(tickfont=dict(color="#aaa", size=11), gridcolor="#1a1a1a"),
        margin=dict(t=10, b=44, l=10, r=90),
        height=240,
        showlegend=False,
        bargap=0.3,
    )
    fig.add_annotation(
        text=f"Base: 50%  →  Prediction: {prob_pct}%",
        xref="paper", yref="paper",
        x=1.0, y=-0.22,
        showarrow=False,
        font=dict(size=10, color="#555"),
        xanchor="right",
    )
    return fig


if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def show_loader(placeholder, step: int):
    """
    step 0 → sadece spinner göster (henüz başlamadı)
    step 1 → PyTorch inference çalışıyor
    step 2 → LangGraph agent çalışıyor
    step 3 → tamamlandı (placeholder temizlenir)
    """
    steps = [
        ("01", "INITIALIZING SENSOR PIPELINE"),
        ("02", "RUNNING PYTORCH INFERENCE"),
        ("03", "DISPATCHING LANGGRAPH AGENT"),
    ]
    icons  = ["◌", "◎", "◌"]
    pct    = [0, 35, 70, 100][step]

    items_html = ""
    for i, (num, label) in enumerate(steps):
        if i + 1 < step:
            cls, icon = "done",   "✓"
        elif i + 1 == step:
            cls, icon = "active", "▶"
        else:
            cls, icon = "",       "◌"
        items_html += f'<li class="avia-step {cls}" data-icon="{icon}">{num} — {label}</li>'

    placeholder.markdown(f"""
<div class="avia-loader-wrap">
    <div class="avia-ring-outer">
        <div class="avia-ring avia-ring-1"></div>
        <div class="avia-ring avia-ring-2"></div>
        <div class="avia-ring avia-ring-3"></div>
        <div class="avia-ring-dot"></div>
    </div>
    <p class="avia-loader-title">DIAGNOSTIC SEQUENCE RUNNING</p>
    <ul class="avia-steps">{items_html}</ul>
    <div class="avia-progress-bar-bg">
        <div class="avia-progress-bar" style="width:{pct}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="AVIA - Intelligent Vehicle Analysis System",
    page_icon="🔧",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

    .main-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; letter-spacing: 2px;
    }
    .subtitle {
        color: #555; font-size: 0.8rem; margin-top: 2px;
        font-family: 'Share Tech Mono', monospace; letter-spacing: 1px;
    }
    .status-ok {
        background: #0d2e1a; border-left: 4px solid #00c851;
        padding: 10px 16px; border-radius: 4px;
        color: #00c851; font-weight: 600; font-size: 0.9rem;
    }
    .status-error {
        background: #2e0d0d; border-left: 4px solid #ff4444;
        padding: 10px 16px; border-radius: 4px;
        color: #ff4444; font-weight: 600;
    }
    .metric-card {
        background: #0f0f1a; border: 1px solid #1e1e2e;
        border-radius: 8px; padding: 16px 20px; text-align: center;
    }
    .metric-label {
        font-size: 0.7rem; color: #555;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px; text-transform: uppercase;
    }
    .metric-value { font-size: 1.5rem; font-weight: 800; margin-top: 4px; }
    .section-title {
        font-size: 0.72rem; color: #444;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 2px; text-transform: uppercase;
        margin-bottom: 8px; border-bottom: 1px solid #1a1a1a; padding-bottom: 6px;
    }

    /* ── Animated Loader ── */
    .avia-loader-wrap {
        background: #0a0a14;
        border: 1px solid #1a1a2e;
        border-radius: 12px;
        padding: 32px 28px;
        text-align: center;
    }

    .avia-ring-outer {
        position: relative;
        width: 88px;
        height: 88px;
        margin: 0 auto 24px;
    }

    .avia-ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 3px solid transparent;
    }
    .avia-ring-1 {
        border-top-color: #00d4ff;
        animation: avia-spin 1.1s linear infinite;
    }
    .avia-ring-2 {
        inset: 10px;
        border-right-color: #0066aa;
        animation: avia-spin 1.6s linear infinite reverse;
    }
    .avia-ring-3 {
        inset: 22px;
        border-bottom-color: #00c851;
        animation: avia-spin 2.2s linear infinite;
    }
    .avia-ring-dot {
        position: absolute;
        inset: 36px;
        border-radius: 50%;
        background: #00d4ff;
        animation: avia-pulse 1.1s ease-in-out infinite;
    }

    @keyframes avia-spin  { to { transform: rotate(360deg); } }
    @keyframes avia-pulse { 0%,100% { opacity: .3; transform: scale(.7); }
                            50%     { opacity: 1;  transform: scale(1);  } }

    .avia-loader-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 2px;
        color: #00d4ff;
        margin-bottom: 20px;
        text-transform: uppercase;
    }

    .avia-steps {
        list-style: none;
        padding: 0;
        margin: 0 0 22px;
        text-align: left;
        display: inline-block;
    }
    .avia-step {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        color: #333;
        padding: 4px 0;
        letter-spacing: 1px;
        transition: color .3s;
    }
    .avia-step.active  { color: #00d4ff; }
    .avia-step.done    { color: #00c851; }
    .avia-step::before { content: attr(data-icon); margin-right: 8px; }

    .avia-progress-bar-bg {
        background: #111122;
        border-radius: 4px;
        height: 4px;
        overflow: hidden;
    }
    .avia-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00d4ff, #00c851);
        border-radius: 4px;
        transition: width .4s ease;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-title">⚙️ AVIA</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AUTONOMOUS VEHICLE INTELLIGENCE AGENT — HYBRID AI DIAGNOSTIC SYSTEM</p>', unsafe_allow_html=True)
st.markdown("---")

if model_loaded:
    st.markdown('<div class="status-ok"> Model and Scaler loaded successfully</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-error"> File not found: {missing_file}<br>Please run train_model.py first!</div>', unsafe_allow_html=True)
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)


left, right = st.columns([1, 1.8], gap="large")

with left:
    st.markdown('<p class="section-title">🎛️ Sensor Inputs</p>', unsafe_allow_html=True)

    air    = st.slider(" Air Temperature (K)",     290.0, 320.0, 300.0, 0.1, help="Normal: 295–305 K")
    proc   = st.slider(" Process Temperature (K)", 300.0, 340.0, 310.0, 0.1, help="Normal: 305–315 K")
    rpm    = st.number_input(" Engine Speed (RPM)", 1000, 5000, 1500, 50,     help="Normal: 1200–2500 RPM")
    torque = st.slider(" Torque (Nm)",               0.0, 100.0,  40.0, 0.5,  help="Normal: 20–60 Nm")
    wear   = st.slider(" Tool Wear (min)",              0,    300,    50,   1, help="Critical above 200 min")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze System", use_container_width=True, type="primary")

with right:
    if analyze_btn:
        if not model_loaded:
            st.error("Model is not loaded!")
        else:
                loader_ph = st.empty()
                try:
                    show_loader(loader_ph, 1)
                    raw = np.array([[air, proc, rpm, torque, wear]], dtype=np.float32)
                    with torch.no_grad():
                        logit = nn_model(torch.FloatTensor(scaler.transform(raw)))
                        prob  = torch.sigmoid(logit).item()

                    prob_pct    = round(prob * 100, 2)
                    is_critical = prob_pct > 50
                    sensor_vals = {
                        "air_temp_K": air, "process_temp_K": proc,
                        "engine_rpm": rpm, "torque_Nm": torque, "wear_min": wear
                    }

                    st.session_state.history.append({
                        "time":        datetime.now().strftime("%H:%M:%S"),
                        "probability": prob_pct,
                        "status":      "CRITICAL" if is_critical else "STABLE",
                        "air_K":       air, "proc_K": proc,
                        "rpm":         rpm, "torque": torque, "wear": wear
                    })

                    show_loader(loader_ph, 2)
                    agent = create_agent()
                    query = (
                        f"Analyze the following sensor data: "
                        f"Air temperature={air}K, Process temperature={proc}K, "
                        f"Engine speed={rpm}RPM, Torque={torque}Nm, Tool wear={wear}min. "
                        f"Use the check_vehicle_health tool and provide a detailed diagnosis."
                    )
                    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
                    llm_text = response["messages"][-1].content

                    show_loader(loader_ph, 3)
                    loader_ph.empty()

                    shap_vals = compute_shap_values(raw)

                    st.session_state.last_result = {
                        "prob_pct":    prob_pct,
                        "is_critical": is_critical,
                        "sensor_vals": sensor_vals,
                        "llm_text":    llm_text,
                        "shap_vals":   shap_vals,
                        "raw_input":   raw,
                    }

                except Exception as e:
                    loader_ph.empty()
                    st.error(f"Agent error: {str(e)}")
                    st.info("Please check your API key or internet connection.")

    if st.session_state.last_result:
        r           = st.session_state.last_result
        prob_pct    = r["prob_pct"]
        is_critical = r["is_critical"]
        sensor_vals = r["sensor_vals"]
        llm_text    = r["llm_text"]
        shap_vals   = r.get("shap_vals")
        raw_input   = r.get("raw_input")

        status_color = "#ff4444" if is_critical else "#00c851"
        status_label = "CRITICAL" if is_critical else "STABLE"
        status_icon  = "🔴" if is_critical else "🟢"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Failure Probability</div>
                <div class="metric-value" style="color:{status_color}">{prob_pct}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">System Status</div>
                <div class="metric-value" style="color:{status_color}">{status_icon} {status_label}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Analyses Run</div>
                <div class="metric-value" style="color:#00d4ff">{len(st.session_state.history)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<p class="section-title">Failure Gauge</p>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(prob_pct), use_container_width=True,
                            config={"displayModeBar": False})
        with g2:
            st.markdown('<p class="section-title">Sensor Radar</p>', unsafe_allow_html=True)
            st.plotly_chart(make_sensor_radar(sensor_vals), use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown('<p class="section-title">Sensor Health Bars</p>', unsafe_allow_html=True)
        st.plotly_chart(make_sensor_bars(sensor_vals), use_container_width=True,
                        config={"displayModeBar": False})

        if shap_vals is not None:
            st.markdown('<p class="section-title">SHAP — Feature Impact</p>', unsafe_allow_html=True)
            st.plotly_chart(make_shap_chart(shap_vals, prob_pct), use_container_width=True,
                            config={"displayModeBar": False})
        elif not shap_available:
            st.caption(" SHAP görselleştirmesi için: `pip install shap`")

        st.markdown('<p class="section-title"> AI Engineer Diagnosis</p>', unsafe_allow_html=True)
        st.chat_message("assistant").write(llm_text)

        if len(st.session_state.history) > 1:
            st.markdown('<p class="section-title"> Probability History</p>', unsafe_allow_html=True)
            st.plotly_chart(make_history_chart(st.session_state.history),
                            use_container_width=True, config={"displayModeBar": False})

            st.markdown('<p class="section-title"> Analysis Log</p>', unsafe_allow_html=True)
            df = pd.DataFrame(st.session_state.history)
            df.columns = ["Time", "Prob %", "Status", "Air K", "Proc K", "RPM", "Torque", "Wear"]
            st.dataframe(
                df.style.map(
                    lambda v: "color: #ff4444" if v == "CRITICAL" else "color: #00c851",
                    subset=["Status"]
                ),
                use_container_width=True,
                hide_index=True
            )

            if st.button(" Clear History", type="secondary"):
                st.session_state.history     = []
                st.session_state.last_result = None
                st.rerun()

    else:
        st.info(" Adjust the sensor values on the left and click **'Analyze System'**.")
        st.markdown("""
        **How does the system work?**
        1. **PyTorch Model** → Analyzes sensor data, calculates failure probability
        2. **Gauge + Radar + Bars** → Visual breakdown of each sensor
        3. **LangGraph Agent** → Interprets model output and makes decisions
        4. **Groq LLM** → Generates engineering-level explanations and recommendations
        5. **History Chart + Log** → Tracks all analyses across the session
        """)


with st.sidebar:
    st.markdown("###  System Info")
    st.markdown("""
    **Model:** DiagnosticNet (PyTorch)  
    **Layers:** 5→32→16→8→1  
    **LLM:** Llama-3.1-8b (Groq)  
    **Orchestration:** LangGraph ReAct  
    """)

    st.markdown("---")
    st.markdown("###  Normal Value Ranges")
    st.markdown("""
    | Sensor | Normal Range |
    |--------|--------------|
    | Air Temperature | 295–305 K |
    | Process Temperature | 305–315 K |
    | Engine Speed | 1200–2500 RPM |
    | Torque | 20–60 Nm |
    | Tool Wear | 0–200 min |
    """)

    st.markdown("---")
    st.markdown("###  Architecture")
    st.code("""
Sensor Data
     ↓
StandardScaler (scaler.pkl)
     ↓
DiagnosticNet (model.pth)
     ↓
Sigmoid → Probability
     ↓
 [Gauge][Radar][Bars]
     ↓
LangGraph ReAct Agent
     ↓
Groq Llama-3.1 LLM
     ↓
Diagnosis + History Log
    """)