# dashboards/pages/p1_model_metrics.py

import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import json
import os
import pandas as pd

dash.register_page(__name__, path="/metrics", name="Model Performance")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
MODEL_NAMES = ["Logistic_Regression", "XGBoost"]

def load_metrics():
    try:
        # Construct path safely
        path = os.path.join(MODEL_DIR, "model_performance_summary.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def summary_table(metrics):
    if not metrics:
        return dbc.Alert("Metrics data file not found in /models/ folder.", color="warning")

    rows = []
    # Filter for metrics that actually exist in your JSON keys
    available_metrics = ["AUC", "KS_Statistic"]
    
    for metric in available_metrics:
        # Create row: Metric Name | Model 1 Value | Model 2 Value
        row_data = [html.Td(metric.replace("_", " "))]
        for m in MODEL_NAMES:
            val = metrics.get(m, {}).get(metric, 0)
            row_data.append(html.Td(f"{val:.4f}"))
        rows.append(html.Tr(row_data))

    return dbc.Table(
        [
            html.Thead(
                html.Tr([html.Th("Metric")] + [html.Th(m.replace("_", " ")) for m in MODEL_NAMES])
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
    )

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H2("Model Performance", className="fw-bold"),
                    html.P(
                        "Predictive accuracy and classification diagnostics",
                        className="text-muted",
                    ),
                ]
            ),
            className="mb-4",
        ),

        dbc.Card(
            [
                dbc.CardHeader("Performance Summary"),
                dbc.CardBody(summary_table(load_metrics())),
            ],
            className="shadow-sm mb-4",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col("ROC Curve", width=6),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="roc-model-select",
                                                options=[
                                                    {
                                                        "label": m.replace("_", " "),
                                                        "value": m,
                                                    }
                                                    for m in MODEL_NAMES
                                                ],
                                                value="XGBoost",
                                                clearable=False,
                                                # REMOVED size="sm" (Invalid argument)
                                                # ADDED style for sizing instead
                                                style={"fontSize": "0.9rem"} 
                                            ),
                                            width=6,
                                        ),
                                    ],
                                    align="center",
                                )
                            ),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="roc-curve-chart",
                                    config={"displayModeBar": False},
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    xs=12,
                    lg=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Confusion Matrix"),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="confusion-matrix-chart",
                                    config={"displayModeBar": False},
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    xs=12,
                    lg=6,
                ),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    className="py-3",
)

@callback(
    Output("roc-curve-chart", "figure"),
    Output("confusion-matrix-chart", "figure"),
    Input("roc-model-select", "value"),
)
def update_charts(model):
    metrics = load_metrics()
    
    # 1. Handle ROC Curve
    roc_path = os.path.join(MODEL_DIR, f"{model.lower()}_roc_data.csv")
    if os.path.exists(roc_path):
        roc_df = pd.read_csv(roc_path)
        auc = metrics[model]["AUC"] if metrics and model in metrics else 0
        roc_fig = px.line(
            roc_df, x="fpr", y="tpr",
            title=f"ROC Curve (AUC = {auc:.4f})",
            template="plotly_white",
        )
        roc_fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey")
        )
    else:
        roc_fig = go.Figure().update_layout(title="ROC Data CSV Not Found")

    # 2. Handle Confusion Matrix
    if metrics and model in metrics:
        m = metrics[model]
        z = [
            [m.get("True_Negatives", 0), m.get("False_Positives", 0)],
            [m.get("False_Negatives", 0), m.get("True_Positives", 0)],
        ]
        cm_fig = px.imshow(
            z,
            x=["Predicted Neg", "Predicted Pos"],
            y=["Actual Neg", "Actual Pos"],
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix: {model.replace('_', ' ')}",
        )
    else:
        cm_fig = go.Figure().update_layout(title="Metrics JSON Not Found")

    return roc_fig, cm_fig