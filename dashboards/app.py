import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import os

# 1. Initialize the app
# use_pages=True looks for files in the 'pages/' folder
app = dash.Dash(
    __name__, 
    use_pages=True, 
    suppress_callback_exceptions=True, 
    external_stylesheets=[dbc.themes.FLATLY]
)

# 2. Expose the Flask server instance for Gunicorn
server = app.server

# 3. Sidebar Layout
sidebar = html.Div([
    html.H2("Credit Risk Monitor", className="display-6 text-primary mb-4"),
    html.Hr(),
    html.P("Free-Data Credit Analytics Pipeline", className="text-muted mb-4"),
    dbc.Nav(
        [
            dbc.NavLink(
                page['name'], 
                href=page['path'], 
                active="exact",
                className="py-2"
            )
            for page in dash.page_registry.values()
        ],
        vertical=True,
        pills=True,
    ),
], className="p-3 bg-light vh-100")

# 4. Main Application Layout
app.layout = dbc.Container([
    dbc.Row([
        # Sidebar (3 columns)
        dbc.Col(sidebar, width=3, className="p-0 border-end"), 
        # Page Content (9 columns)
        dbc.Col(dash.page_container, width=9, id="page-content", className="p-4")
    ])
], fluid=True)

# 5. Production Entry Point
if __name__ == '__main__':
    # On Render, the 'PORT' environment variable is used to set the port
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)