# dashboards/pages/p2_interpretability.py
# -*- coding: utf-8 -*-
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pandas as pd

dash.register_page(__name__, path="/explain", name="Interpretability & SHAP")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))

def load_coefficients():
    path = os.path.join(MODEL_DIR, "log_reg_coefficients.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_shap_data():
    path = os.path.join(MODEL_DIR, "xgboost_shap_values.csv")
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def create_coefficient_chart(coeffs):
    if not coeffs:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    text="No Coefficients Found",
                    showarrow=False,
                )
            ]
        )
        return fig

    df = pd.DataFrame(coeffs.items(), columns=["Feature", "Coefficient"])
    df["Abs_Coeff"] = df["Coefficient"].abs()
    df = df.sort_values("Abs_Coeff", ascending=True)

    fig = px.bar(
        df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale=px.colors.diverging.RdBu,
        title="Feature Impact",
    )
    fig.update_layout(showlegend=False, template="plotly_white")
    return fig

def create_shap_summary_chart(shap_df):
    if shap_df is None or shap_df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    text="No SHAP Data Found",
                    showarrow=False,
                )
            ]
        )
        return fig

    features_only = shap_df.drop(columns=["expected_value"], errors="ignore")
    mean_abs = features_only.abs().mean().sort_values(ascending=False)
    df_imp = pd.DataFrame(
        {
            "Feature": mean_abs.index,
            "Mean_Abs_SHAP": mean_abs.values,
        }
    )

    fig = px.bar(
        df_imp,
        x="Mean_Abs_SHAP",
        y="Feature",
        orientation="h",
        title="XGBoost SHAP Importance",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        template="plotly_white",
    )
    return fig

layout = html.Div(
    [
        html.H1("Model Interpretability"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            figure=create_coefficient_chart(
                                load_coefficients()
                            ),
                            style={"height": "500px"},
                        )
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="shap-summary-chart",
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
    Output("shap-summary-chart", "figure"),
    Input("shap-summary-chart", "id"),
)
def update_shap_chart(_):
    return create_shap_summary_chart(load_shap_data())
