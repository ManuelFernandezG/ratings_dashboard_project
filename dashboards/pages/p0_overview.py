# dashboards/pages/p0_overview.py
# -*- coding: utf-8 -*-

import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import os

dash.register_page(__name__, path="/", name="Overview")

# --- Styling Constants ---
DARK_GRAY = "#2c3e50"  # Professional Slate Dark Gray

# --- Data Loading Logic ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "company_fundamentals.csv")

try:
    df = pd.read_csv(DATA_PATH)
    if {"date", "symbol"}.issubset(df.columns):
        latest_df = (
            df.sort_values("date", ascending=False)
            .drop_duplicates(subset="symbol")
        )
    else:
        latest_df = df
except Exception:
    latest_df = pd.DataFrame(columns=["Sector", "Region", "symbol"])

# --- Helper Functions ---
def exposure_chart(df, col, title):
    if df.empty or col not in df.columns:
        return px.bar(title=f"{title} (No Data)")
    
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    
    fig = px.bar(
        counts,
        x=col,
        y="Count",
        title=title,
        template="plotly_white",
        color_discrete_sequence=[DARK_GRAY]
    )
    fig.update_layout(
        margin=dict(t=60, l=20, r=20, b=20),
        xaxis_title=None,
        yaxis_title="Number of Companies"
    )
    return fig

# --- Layout Definition ---
layout = dbc.Container(
    [
        # 1. Header Row
        dbc.Row(
            dbc.Col(
                [
                    html.H2("Credit Risk Overview", className="fw-bold text-primary"),
                    html.Hr(),
                    html.P(
                        "Automated forward-looking credit risk evaluation combining financial fundamentals with macro-financial indicators.",
                        className="lead",
                    ),
                ],
                xs=12,
            ),
            className="mb-4",
        ),

        # 2. Detailed Interpretation Section
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H4("1. Data, Model, and Results — Detailed Interpretation", className="mb-0")),
                        dbc.CardBody([
                            # 1.1 Objective
                            html.H5("1.1 Objective of the Analysis", className="fw-bold text-secondary"),
                            html.P(
                                "The goal is to predict systemic credit stress events by merging company-level fundamentals "
                                "with macroeconomic data. A 'High-Risk Event' is defined as:"
                            ),
                            dbc.Alert(
                                "A widening of the ICE BofA US High Yield Option-Adjusted Spread (HY OAS) by ≥ 500 basis points within the subsequent 90 days.",
                                color="danger", className="fw-bold"
                            ),
                            
                            html.Hr(),

                            # 1.2 Data Sources
                            html.H5("1.2 Data Sources and Feature Construction", className="fw-bold text-secondary"),
                            
                            html.H6("A. Macroeconomic & Market Indicators (Daily)", className="mt-3"),
                            dbc.Table([
                                html.Thead(html.Tr([html.Th("Feature"), html.Th("Description"), html.Th("Economic Meaning")])),
                                html.Tbody([
                                    html.Tr([html.Td("IG_OAS"), html.Td("ICE BofA Investment Grade OAS"), html.Td("Early signal of broad credit risk repricing")]),
                                    html.Tr([html.Td("HIGH_YIELD_OAS"), html.Td("ICE BofA High Yield OAS"), html.Td("Current level of speculative-grade risk")]),
                                    html.Tr([html.Td("FED_FUNDS"), html.Td("Effective Federal Funds Rate"), html.Td("Monetary policy stance")]),
                                    html.Tr([html.Td("TED_SPREAD"), html.Td("LIBOR – T-bill spread"), html.Td("Banking system and funding stress")]),
                                    html.Tr([html.Td("VIX"), html.Td("Implied equity volatility"), html.Td("Market risk aversion")]),
                                ])
                            ], bordered=True, size="sm", hover=True),

                            html.H6("B. Company Fundamentals (Annual, Forward-Filled)", className="mt-3"),
                            dbc.Table([
                                html.Thead(html.Tr([html.Th("Feature"), html.Th("Description"), html.Th("Rationale")])),
                                html.Tbody([
                                    html.Tr([html.Td("PROFIT_MARGIN"), html.Td("Net income / revenue"), html.Td("Earnings resilience")]),
                                    html.Tr([html.Td("LIQUIDITY_RATIO"), html.Td("Current assets / current liabilities"), html.Td("Short-term solvency")]),
                                    html.Tr([html.Td("LEVERAGE"), html.Td("Debt / equity or assets"), html.Td("Balance-sheet risk")]),
                                ])
                            ], bordered=True, size="sm", hover=True),

                            html.H6("C. Target Variable (Label)", className="mt-3"),
                            html.P([
                                html.B("1 (High Risk): "), "HY OAS widens ≥ 500 bps within 90 days. ",
                                html.Br(),
                                html.B("0 (Low Risk): "), "No such widening. ",
                                html.Br(),
                                html.Small("This overlapping-window approach ensures each daily observation forecasts a 90-day forward horizon.", className="text-muted")
                            ]),

                            html.Hr(),

                            # 1.3 Model Types
                            html.H5("1.3 Model Types Used", className="fw-bold text-secondary"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Strong("Logistic Regression"),
                                        html.P("Linear and interpretable. Used for understanding the economic 'why' behind risk changes.")
                                    ], className="p-2 border rounded bg-light")
                                ], md=6),
                                dbc.Col([
                                    html.Div([
                                        html.Strong("XGBoost (Gradient-Boosting)"),
                                        html.P("Nonlinear and interaction-aware. Used for maximum predictive accuracy and early warning.")
                                    ], className="p-2 border rounded bg-light")
                                ], md=6),
                            ]),
                        ]),
                    ],
                    className="shadow-sm mb-5",
                ),
                xs=12,
            ),
        ),

        # 3. Graphs Section (Keeping Only Sector Exposure)
        html.H4("Portfolio Exposure Snapshot", className="fw-bold mb-3"),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Sector Exposure"),
                        dbc.CardBody(
                            dcc.Graph(
                                figure=exposure_chart(latest_df, "Sector", "Portfolio Distribution by Sector"),
                                config={"displayModeBar": False},
                            )
                        ),
                    ],
                    className="shadow-sm mb-5",
                ),
                xs=12,
            ),
        ),
    ],
    fluid=True,
    className="py-3",
)