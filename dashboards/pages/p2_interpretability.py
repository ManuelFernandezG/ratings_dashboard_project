# -*- coding: utf-8 -*-
"""
dashboards/pages/p2_interpretability.py

Page 2: Model explainability using Logistic Regression Coefficients and SHAP.
"""
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pandas as pd
import numpy as np

dash.register_page(__name__, path='/explain', name='Interpretability & SHAP')

# --- 1. PATH-PROOF DIRECTORY SETUP ---
# Finds the 'models' folder by going up two levels from 'dashboards/pages'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))

XGBOOST_NAME = "XGBoost"

# --- 2. DATA LOADING FUNCTIONS ---

def load_coefficients():
    """Loads Logistic Regression coefficients from JSON file."""
    path = os.path.join(MODEL_DIR, "log_reg_coefficients.json")
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"DEBUG: Coefficients not found at {path}")
        return None

def load_shap_data():
    """Loads XGBoost SHAP values from CSV."""
    path = os.path.join(MODEL_DIR, f"{XGBOOST_NAME.lower()}_shap_values.csv")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"DEBUG: SHAP values not found at {path}")
        return None

# --- 3. CHART CREATION FUNCTIONS ---

def create_coefficient_chart(coefficients):
    """Creates a horizontal bar chart of Logistic Regression coefficients."""
    if coefficients is None:
        return go.Figure().update_layout(
            annotations=[dict(text="Logistic Regression Coefficients not found.", showarrow=False)]
        )
    
    df = pd.DataFrame(coefficients.items(), columns=['Feature', 'Coefficient'])
    df['Abs_Coeff'] = df['Coefficient'].abs()
    df = df.sort_values('Abs_Coeff', ascending=True)

    fig = px.bar(
        df, 
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale=px.colors.diverging.RdBu,
        color_continuous_midpoint=0,
        title="Logistic Regression Coefficients (Feature Impact)",
        labels={'Coefficient': 'Coefficient Value'},
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), template='plotly_white')
    return fig

def create_shap_summary_chart(shap_df):
    """Creates a SHAP Feature Importance bar chart."""
    if shap_df is None:
        return go.Figure().update_layout(
            annotations=[dict(text="XGBoost SHAP data not found. Run evaluation step.", showarrow=False)]
        )

    # Exclude expected_value column and calculate mean absolute SHAP
    features_only = shap_df.drop(columns=['expected_value'], errors='ignore')
    mean_abs_shap = features_only.abs().mean().sort_values(ascending=False)
    
    df_importance = pd.DataFrame({
        'Feature': mean_abs_shap.index,
        'Mean_Abs_SHAP': mean_abs_shap.values
    })

    fig = px.bar(
        df_importance,
        x='Mean_Abs_SHAP',
        y='Feature',
        orientation='h',
        title="XGBoost Feature Importance (Mean Absolute SHAP)",
        labels={'Mean_Abs_SHAP': 'Impact on Model Output'},
        color_discrete_sequence=['#4C78A8']
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=40, b=10), template='plotly_white')
    
    return fig

# --- 4. LAYOUT DEFINITION ---

layout = html.Div([
    html.H1("Model Interpretability: Coefficients & SHAP"),
    html.Hr(),

    dbc.Row([
        # Logistic Regression Column
        dbc.Col([
            html.H3("Logistic Regression: Feature Coefficients"),
            dcc.Graph(
                figure=create_coefficient_chart(load_coefficients()),
                style={'height': '500px'}
            ),
        ], md=6),

        # SHAP Column
        dbc.Col([
            html.H3("XGBoost: SHAP Summary"),
            dcc.Graph(
                id='shap-summary-chart',
                style={'height': '500px'} 
            )
        ], md=6),
    ], className="g-4"),
])

# --- 5. CALLBACKS ---

@callback(
    Output('shap-summary-chart', 'figure'),
    [Input('shap-summary-chart', 'id')] 
)
def update_shap_chart(_):
    shap_df = load_shap_data()
    return create_shap_summary_chart(shap_df)