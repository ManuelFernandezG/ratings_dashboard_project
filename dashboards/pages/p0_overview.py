# -*- coding: utf-8 -*-
"""
dashboards/pages/p0_overview.py

Page 0: Dashboard Overview, Synthetic Rating Distribution, and Exposure Breakdown.
"""
# -*- coding: utf-8 -*-
"""
dashboards/pages/p0_overview.py

Page 0: Dashboard Overview, Synthetic Rating Distribution, and Exposure Breakdown.
"""

import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# 1. Correct Page Registration
dash.register_page(__name__, path='/', name='Credit Dashboard')

# 2. Correct Data Path and Processing
# We look up one level from 'pages' to find the 'data' folder
try:
    df = pd.read_csv("../data/raw/company_fundamentals.csv")
    
    # Process data to get the latest entry for each company symbol
    if 'date' in df.columns and 'symbol' in df.columns:
        latest_df = df.sort_values('date', ascending=False).drop_duplicates(subset=['symbol'])
    else:
        latest_df = df
        
except FileNotFoundError:
    print("ERROR: Processed data file not found. Ensure src/ingestion/load_fundamentals.py was run.")
    latest_df = pd.DataFrame(columns=['Sector', 'Region', 'symbol'])

# 3. Create the Exposure Charts Function
def create_exposure_chart(df, column, title):
    if df.empty or column not in df.columns:
        # Return an empty figure with a clear title if data is missing
        fig = px.bar(title=f"{title} (No Data Found)")
        fig.update_layout(template='plotly_white')
        return fig
    
    counts = df.groupby(column).size().reset_index(name='count')
    fig = px.bar(counts, x=column, y='count', title=title, template='plotly_white')
    return fig

# 4. Layout Definition
layout = dbc.Container([
    html.H1("Dashboard Overview & Portfolio Exposure"),
    html.Hr(),
    
    # Top Row: Synthetic Risk Distribution
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                figure=px.histogram(
                    latest_df, 
                    x="Sector", 
                    title="Synthetic Risk Distribution (by Sector)"
                ).update_layout(template='plotly_white'),
                id='rating-distribution-chart'
            )
        ], md=12),
    ], className="mb-4"),

    # Bottom Row: Sector and Region Breakdown
    dbc.Row([
        # Sector Exposure Chart
        dbc.Col([
            dcc.Graph(
                figure=create_exposure_chart(latest_df, 'Sector', 'Sector Exposure'),
                id='sector-exposure-chart'
            )
        ], md=6),
        
        # Region Exposure Chart
        dbc.Col([
            dcc.Graph(
                figure=create_exposure_chart(latest_df, 'Region', 'Region Exposure'),
                id='region-exposure-chart'
            )
        ], md=6),
    ])
], fluid=True, className="p-3")