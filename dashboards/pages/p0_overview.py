# dashboards/pages/p0_overview.py
# -*- coding: utf-8 -*-
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import os

dash.register_page(__name__, path='/', name='Credit Dashboard')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "company_fundamentals.csv")

try:
    df = pd.read_csv(DATA_PATH)
    if "date" in df.columns and "symbol" in df.columns:
        latest_df = (
            df.sort_values("date", ascending=False)
            .drop_duplicates(subset=["symbol"])
        )
    else:
        latest_df = df
except Exception:
    latest_df = pd.DataFrame(columns=["Sector", "Region", "symbol"])

def create_exposure_chart(dataframe, column, title):
    if dataframe.empty or column not in dataframe.columns:
        fig = px.bar(title=f"{title} (No Data Found)")
        fig.update_layout(template="plotly_white")
        return fig
    counts = dataframe.groupby(column).size().reset_index(name="count")
    fig = px.bar(
        counts,
        x=column,
        y="count",
        title=title,
        template="plotly_white"
    )
    return fig

layout = dbc.Container(
    [
        html.H1("Dashboard Overview & Portfolio Exposure"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="rating-distribution-chart",
                            figure=px.histogram(
                                latest_df,
                                x="Sector",
                                title="Synthetic Risk Distribution",
                                template="plotly_white",
                            ),
                        )
                    ],
                    md=12,
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="sector-exposure-chart",
                            figure=create_exposure_chart(
                                latest_df,
                                "Sector",
                                "Sector Exposure",
                            ),
                        )
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="region-exposure-chart",
                            figure=create_exposure_chart(
                                latest_df,
                                "Region",
                                "Region Exposure",
                            ),
                        )
                    ],
                    md=6,
                ),
            ]
        ),
    ],
    fluid=True,
    className="p-3",
)
