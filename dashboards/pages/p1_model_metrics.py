"""
dashboards/pages/p1_model_metrics.py

Page 1: Model performance comparison (AUC, ROC, KS, Confusion Matrices).
"""
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pandas as pd
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/metrics', name='Model Performance')

# --- 1. PATH-PROOF DIRECTORY SETUP ---
# This finds the 'models' folder regardless of where you run the script from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))

MODEL_NAMES = ["Logistic_Regression", "XGBoost"]

# --- 2. DATA LOADING FUNCTIONS ---

def load_metrics():
    summary_path = os.path.join(MODEL_DIR, "model_performance_summary.json")
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"DEBUG: File not found at {summary_path}")
        return None

def create_summary_table(metrics):
    if metrics is None:
        return html.P("Performance summary not found. Ensure models/model_performance_summary.json exists.")

    models = list(metrics.keys())
    data = [
        ['Metric'] + models,
        ['AUC'] + [f"{metrics[m]['AUC']:.4f}" for m in models],
        ['KS Statistic'] + [f"{metrics[m]['KS_Statistic']:.4f}" for m in models],
        ['True Positives (TP)'] + [f"{metrics[m].get('True_Positives', 'N/A')}" for m in models],
        ['False Negatives (FN)'] + [f"{metrics[m].get('False_Negatives', 'N/A')}" for m in models],
    ]

    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in data[0]])),
            html.Tbody([html.Tr([html.Td(val) for val in row]) for row in data[1:]])
        ],
        bordered=True, hover=True, responsive=True, striped=True, className="mt-3"
    )

# --- 3. LAYOUT DEFINITION ---

layout = html.Div([
    html.H1("Model Comparison: AUC, KS & Confusion Matrices 🔬"),
    html.Hr(),

    html.H3("Performance Summary"),
    create_summary_table(load_metrics()),
    html.Br(),

    dbc.Row([
        # ROC Curve Column
        dbc.Col([
            html.H3("ROC Curve"),
            dcc.Dropdown(
                id='roc-model-select',
                options=[{'label': name.replace('_', ' '), 'value': name} for name in MODEL_NAMES],
                value='XGBoost',
                clearable=False
            ),
            dcc.Graph(id='roc-curve-chart', style={'height': '500px'})
        ], md=6),
        
        # Confusion Matrix Column (FIXED: Now has a Graph component)
        dbc.Col([
            html.H3("Confusion Matrix (Threshold 0.5)"),
            dcc.Graph(id='confusion-matrix-chart', style={'height': '500px'})
        ], md=6),
    ], className="g-4"),
])

# --- 4. CALLBACKS ---

@callback(
    [Output('roc-curve-chart', 'figure'),
     Output('confusion-matrix-chart', 'figure')],
    [Input('roc-model-select', 'value')]
)
def update_performance_charts(model_name):
    metrics = load_metrics()
    
    # --- ROC CURVE LOGIC ---
    roc_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_roc_data.csv")
    if not os.path.exists(roc_path):
        roc_fig = go.Figure().update_layout(title="ROC Data Not Found")
    else:
        df_roc = pd.read_csv(roc_path)
        auc_val = metrics[model_name]['AUC'] if metrics else 0
        roc_fig = px.line(df_roc, x='fpr', y='tpr', title=f'ROC Curve: {model_name} (AUC={auc_val:.4f})')
        roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

    # --- CONFUSION MATRIX LOGIC ---
    if not metrics or model_name not in metrics:
        cm_fig = go.Figure().update_layout(title="Confusion Matrix Data Not Found")
    else:
        m = metrics[model_name]
        # Build matrix: [[TN, FP], [FN, TP]]
        z = [[m['True_Negatives'], m['False_Positives']], 
             [m['False_Negatives'], m['True_Positives']]]
        
        cm_fig = px.imshow(
            z,
            text_auto=True,
            x=['Predicted Low Risk', 'Predicted High Risk'],
            y=['Actual Low Risk', 'Actual High Risk'],
            labels=dict(x="Prediction", y="Actual Value"),
            color_continuous_scale='Blues',
            title=f"Confusion Matrix: {model_name}"
        )

    return roc_fig, cm_fig