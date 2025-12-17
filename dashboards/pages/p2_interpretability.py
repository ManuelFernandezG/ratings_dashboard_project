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

# Styling Constants
DARK_GRAY = "#2c3e50"
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
        fig.update_layout(annotations=[dict(text="No Coefficients Found", showarrow=False)])
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
        title="1.4 Logistic Regression: Feature Impact"
    )
    fig.update_layout(showlegend=False, template="plotly_white", coloraxis_showscale=False)
    return fig

def create_shap_summary_chart(shap_df):
    if shap_df is None or shap_df.empty:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No SHAP Data Found", showarrow=False)])
        return fig

    features_only = shap_df.drop(columns=["expected_value"], errors="ignore")
    mean_abs = features_only.abs().mean().sort_values(ascending=False)
    df_imp = pd.DataFrame({"Feature": mean_abs.index, "Mean_Abs_SHAP": mean_abs.values})

    fig = px.bar(
        df_imp,
        x="Mean_Abs_SHAP",
        y="Feature",
        orientation="h",
        title="XGBoost SHAP Importance",
        color_discrete_sequence=[DARK_GRAY]
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, template="plotly_white")
    return fig

layout = dbc.Container([
    html.H1("Model Interpretability", className="fw-bold text-primary mt-3"),
    html.Hr(),
    
    # 1.4 & 1.5 Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("1.4 Coefficient Interpretation", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_coefficient_chart(load_coefficients()),
                        style={"height": "400px"},
                        config={"displayModeBar": False}
                    ),
                    html.Div([
                        html.P([html.B("Legend: "), "Blue increases probability of high-risk event | Red decreases probability."]),
                        html.Ul([
                            html.Li([html.B("IG_OAS: "), "Strongest impact. Stress emerges in high-quality credit first."]),
                            html.Li([html.B("TED_SPREAD: "), "Reflects acute liquidity stress phases."]),
                            html.Li([html.B("FED_FUNDS: "), "Impact is primary through spreads, not direct."]),
                            html.Li([html.B("Fundamentals: "), "Minimal impact on systemic market-wide shocks."]),
                        ])
                    ], className="small mt-2")
                ])
            ], className="shadow-sm mb-4")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("1.5 Performance & Comparison", className="mb-0")),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead(html.Tr([html.Th("Metric"), html.Th("LogReg"), html.Th("XGBoost")])),
                        html.Tbody([
                            html.Tr([html.Td("AUC"), html.Td("0.9360"), html.Td("0.9996")]),
                            html.Tr([html.Td("KS Stat"), html.Td("0.7500"), html.Td("0.9970")]),
                            html.Tr([html.Td("True Positives"), html.Td("262"), html.Td("312")]),
                            html.Tr([html.Td("False Negatives"), html.Td("50"), html.Td("0")]),
                        ])
                    ], bordered=True, size="sm", className="text-center"),
                    html.H6("1.6 Key Modeling Insight", className="fw-bold mt-3"),
                    html.P("LogReg provides economic narrative; XGBoost excels at timing/precision.", className="small"),
                    dcc.Graph(
                        id="shap-summary-chart",
                        style={"height": "300px"},
                        config={"displayModeBar": False}
                    )
                ])
            ], className="shadow-sm mb-4")
        ], md=6),
    ]),

    # 2. Executive Summary
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("2. Executive Summary", className="mb-0 text-white bg-dark")),
                dbc.CardBody([
                    html.P(
                        "Investment-grade credit spreads are the most powerful leading indicator of future high-yield market stress. "
                        "Systemic credit risk is driven primarily by macro-financial conditions rather than firm-specific metrics.",
                        className="fw-bold"
                    ),
                    html.P(
                        "The XGBoost model significantly outperforms the linear model by capturing nonlinear interactions, achieving "
                        "perfect recall (0 False Negatives) compared to the Logistic Regression model (50 False Negatives). "
                        "Together, these models form a robust framework for proactive monitoring and decision support."
                    )
                ])
            ], className="shadow mb-5")
        )
    )
], fluid=True)

@callback(
    Output("shap-summary-chart", "figure"),
    Input("shap-summary-chart", "id"),
)
def update_shap_chart(_):
    return create_shap_summary_chart(load_shap_data())