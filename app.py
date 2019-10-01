import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle

########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

########### Set up the layout
app.layout = html.Div(children=[
    html.H1('This is the headline'),
    html.Div(['Stuff will go here!']),
    html.Br(),
    html.A('Code on Github', href='https://github.com/austinlasseter/knn_iris_plotly'),
])

############ Execute the app
if __name__ == '__main__':
    app.run_server()
