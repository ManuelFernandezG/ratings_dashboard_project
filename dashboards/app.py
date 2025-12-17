# dashboards/app.py
# -*- coding: utf-8 -*-

import dash
from dash import html
import dash_bootstrap_components as dbc
import os

# ---------------------------------------------------------
# Initialize Dash App
# ---------------------------------------------------------

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, shrink-to-fit=no",
        }
    ],
)

# Expose Flask server (for Gunicorn / Render)
server = app.server

# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------

def build_sidebar():
    return html.Div(
        [
            html.Div(
                [
                    html.H4(
                        "Credit Risk",
                        className="fw-bold mb-0 text-primary",
                    ),
                    html.Small(
                        "Analytics Platform",
                        className="text-muted",
                    ),
                ],
                className="mb-4",
            ),

            dbc.Nav(
                [
                    dbc.NavLink(
                        page["name"],
                        href=page["path"],
                        active="exact",
                        className="mb-1",
                    )
                    for page in dash.page_registry.values()
                ],
                vertical=True,
                pills=True,
            ),

            html.Hr(),

            html.Small(
                "by Manuel Fernandez",
                className="text-muted",
            ),
        ],
        className="p-3 bg-light h-100",
    )


# ---------------------------------------------------------
# App Layout
# ---------------------------------------------------------

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    build_sidebar(),
                    xs=12,
                    md=4,
                    lg=3,
                    className="border-end",
                ),

                # Main Page Content
                dbc.Col(
                    dash.page_container,
                    xs=12,
                    md=8,
                    lg=9,
                    className="bg-white",
                ),
            ],
            className="g-0 min-vh-100",
        )
    ],
    fluid=True,
)


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
