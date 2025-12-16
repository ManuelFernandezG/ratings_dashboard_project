import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# 1. Initialize the app first (instantiation)
app = dash.Dash(
    __name__, 
    use_pages=True, # Enables multi-page support
    suppress_callback_exceptions=True, 
    external_stylesheets=[dbc.themes.FLATLY]
)

# 2. Sidebar Layout
sidebar = html.Div([
    html.H2("Credit Risk Monitor", className="display-4 text-primary"),
    html.Hr(),
    html.P("Free-Data Credit Analytics Pipeline", className="lead"),
    html.Div([
        # This automatically generates links for files in the /pages folder
        dcc.Link(f" {page['name']}", href=page['path'], className="nav-link")
        for page in dash.page_registry.values()
    ], className="nav flex-column"),
], className="p-3 bg-light h-100")

# 3. Main Application Layout
app.layout = dbc.Container([
    dbc.Row([
        # Sidebar (3 columns)
        dbc.Col(sidebar, width=3, className="bg-light"), 
        # Page Content (9 columns) - This is where your pages will load
        dbc.Col(dash.page_container, width=9, id="page-content", className="p-4")
    ])
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)