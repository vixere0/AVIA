# AVIA
AVIA (Autonomous Vehicle Intelligence Agent) is an advanced diagnostic framework designed for predictive maintenance in industrial and autonomous systems. The project bridges the gap between high-performance "black-box" models and actionable engineering decisions by integrating deep learning with agentic workflows. 
# ⚙️ AVIA: Autonomous Vehicle Intelligence Agent

**AVIA** is an advanced hybrid diagnostic system designed for predictive maintenance in industrial and autonomous vehicle environments. It bridges the gap between traditional machine learning and modern **Agentic AI workflows**[cite: 4].

##  Key Features
*   **Hybrid Architecture**: Integrates a custom **PyTorch Neural Network** (MLP) with a **LangGraph ReAct Agent** for multi-step logical reasoning[cite: 3, 4].
*   **Explainable AI (XAI)**: Features **SHAP (SHapley Additive exPlanations)** integration to provide feature-level transparency for model predictions[cite: 4].
*   **Real-time Diagnostics**: Processes sensor fusion data (Temperature, RPM, Torque, Wear) to detect system failures with optimized recall[cite: 3, 4].
*   **Intelligent Reasoning**: Leverages **Groq-powered Llama 3.1** to translate complex model outputs into human-readable engineering reports[cite: 4].
*   **Interactive Dashboard**: A professional **Streamlit** UI including radar charts, failure gauges, and real-time historical logging[cite: 4].

##  Tech Stack
*   **Frameworks**: PyTorch, LangGraph, LangChain[cite: 4]
*   **LLM**: Llama 3.1 8B (via Groq API)[cite: 4]
*   **Interpretability**: SHAP[cite: 4]
*   **Visualization**: Streamlit, Plotly[cite: 4]
*   **Data Science**: Pandas, Scikit-learn, Joblib[cite: 3, 4]

##  Project Structure
*   `train_model.py`: End-to-end training pipeline for the DiagnosticNet model[cite: 3].
*   `agent.py`: Main application script featuring the hybrid agent logic[cite: 4].
*   `models/`: Pre-trained weights (`diagnostic_model.pth`) and fitted scalers (`scaler.pkl`)[cite: 2, 3].
*   `data/`: Industrial sensor dataset (`ai4i2020.csv`)[cite: 3].

##  Quick Start
1. Install dependencies: `pip install -r requirements.txt`[cite: 4]
2. Add your `GROQ_API_KEY` to a `.env` file[cite: 4].
3. Launch the agent: `streamlit run agent.py`[cite: 4]
