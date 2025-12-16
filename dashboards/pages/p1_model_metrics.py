# dashboards/pages/p1_model_metrics.py
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pandas as pd
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/metrics", name="Model Performance")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
MODEL_NAMES = ["Logistic_Regression", "XGBoost"]

def load_metrics():
    path = os.path.join(MODEL_DIR, "model_performance_summary.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def create_summary_table(metrics):
    if not metrics:
        return html.P("Performance summary not found.")
    models = list(metrics.keys())
    header = ["Metric"] + models
    rows = [
        ["AUC"] + [f"{metrics[m]['AUC']:.4f}" for m in models],
        ["KS Statistic"] + [f"{metrics[m]['KS_Statistic']:.4f}" for m in models],
        ["True Positives"] + [metrics[m].get("True_Positives", "N/A") for m in models],
        ["False Negatives"] + [metrics[m].get("False_Negatives", "N/A") for m in models],
    ]
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in header])),
            html.Tbody(
                [html.Tr([html.Td(cell) for cell in row]) for row in rows]
            ),
        ],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        className="mt-3",
    )

layout = html.Div(
    [
        html.H1("Model Comparison 🔬"),
        html.Hr(),
        create_summary_table(load_metrics()),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="roc-model-select",
                            options=[
                                {"label": n.replace("_", " "), "value": n}
                                for n in MODEL_NAMES
                            ],
                            value="XGBoost",
                            clearable=False,
                        ),
                        dcc.Graph(
                            id="roc-curve-chart",
                            style={"height": "500px"},
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="confusion-matrix-chart",
                            style={"height": "500px"},
                        )
                    ],
                    md=6,
                ),
            ],
            className="g-4",
        ),
    ]
)

@callback(
    Output("roc-curve-chart", "figure"),
    Output("confusion-matrix-chart", "figure"),
    Input("roc-model-select", "value"),
)
def update_performance_charts(model_name):
    metrics = load_metrics()

    roc_path = os.path.join(
        MODEL_DIR,
        f"{model_name.lower()}_roc_data.csv",
    )

    if os.path.exists(roc_path):
        roc_df = pd.read_csv(roc_path)
        auc_val = metrics[model_name]["AUC"] if metrics else 0
        roc_fig = px.line(
            roc_df,
            x="fpr",
            y="tpr",
            title=f"ROC Curve (AUC={auc_val:.4f})",
        )
        roc_fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(dash="dash"),
        )
    else:
        roc_fig = go.Figure()
        roc_fig.update_layout(title="ROC Data Not Found")

    if metrics and model_name in metrics:
        m = metrics[model_name]
        z = [
            [m["True_Negatives"], m["False_Positives"]],
            [m["False_Negatives"], m["True_Positives"]],
        ]
        cm_fig = px.imshow(
            z,
            text_auto=True,
            x=["Low Risk", "High Risk"],
            y=["Actual Low", "Actual High"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix: {model_name}",
        )
    else:
        cm_fig = go.Figure()
        cm_fig.update_layout(title="Matrix Data Missing")

    return roc_fig, cm_fig
