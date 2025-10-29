import os
import pandas as pd
import plotly_express as px
import dash
from dash.dependencies import Input, Output
from dash import dash_table, dcc, html
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Binarizer




data = pd.read_csv("dane.csv", delimiter=';', index_col=0)
# print(data)

# Remove spaces from strings
data = data.replace('\s', '', regex=True)

# Replace NaN with empty string
data = data.fillna('')

# Remove spaces from column values
data = data.apply(lambda x: x.str.replace(' ', ''))

# Replace commas with dots for decimal numbers
data = data.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)

# Convert all values to numeric
data = data.apply(pd.to_numeric)

# ARRIVALS DATA
df_arrivals = data.iloc[0] # row related to arrivals
df_arrivals = df_arrivals.iloc[1:] # columns with years
df_arrivals = df_arrivals.apply(pd.to_numeric)

df_overnights = data.iloc[1]
df_overnights = df_overnights.iloc[1:]
df_overnights = df_overnights.apply(pd.to_numeric)

df_sameday = data.iloc[2]
df_sameday = df_sameday.iloc[1:]
df_sameday = df_sameday.apply(pd.to_numeric)

# DEPARTURES DATA
df_departure = data.iloc[16, 1:].apply(pd.to_numeric)
df_dep_overnights = data.iloc[17, 1:].apply(pd.to_numeric)
df_dep_sameday = data.iloc[18, 1:].apply(pd.to_numeric)

# INBOUND TOURISM EXPENDITURE
df_inb_expend = data.iloc[3, 1:].apply(pd.to_numeric)
df_travel_inb = data.iloc[4, 1:].apply(pd.to_numeric)
df_transport_inb = data.iloc[5, 1:].apply(pd.to_numeric)

# OUTBOUND TOURISM EXPENDITURE
df_out_expend = data.iloc[19, 1:].apply(pd.to_numeric)
df_travel_out = data.iloc[20, 1:].apply(pd.to_numeric)
df_transport_out = data.iloc[21, 1:].apply(pd.to_numeric)

# Top countries Poles travel to
df_Croatia = data.iloc[10, 1:].apply(pd.to_numeric)
df_Czech  = data.iloc[11, 1:].apply(pd.to_numeric)
df_Germany = data.iloc[12, 1:].apply(pd.to_numeric)
df_Italy = data.iloc[13, 1:].apply(pd.to_numeric)
df_UK = data.iloc[14, 1:].apply(pd.to_numeric)
df_Greece = data.iloc[15, 1:].apply(pd.to_numeric)

years = data.columns[1:].astype(int)

# ARRIVALS CHART
fig_combined = go.Figure()

fig_combined.add_trace(go.Scatter(x=df_arrivals.index, y=df_arrivals.values, mode='lines+markers', name='Total arrivals'))
fig_combined.add_trace(go.Scatter(x=df_overnights.index, y=df_overnights.values, mode='lines+markers', name='Overnights visitors'))
fig_combined.add_trace(go.Scatter(x=df_sameday.index, y=df_sameday.values, mode='lines+markers', name='Same-day visitors'))

fig_combined.update_layout(xaxis_title='Year', yaxis_title='Number of arrivals')

# Percentage change in arrivals/departures
rate_arrivals = df_arrivals.pct_change()*100
rate_arrivals = rate_arrivals[1:] # first value has no data

rate_departure = df_departure.pct_change()*100
rate_departure = rate_departure[1:] # first value has no data


# DEPARTURES CHART
fig_departure = go.Figure()

fig_departure.add_trace(go.Scatter(x=df_departure.index, y=df_departure.values, mode='lines+markers', name='Total departure'))
fig_departure.add_trace(go.Scatter(x=df_dep_overnights.index, y=df_dep_overnights.values, mode='lines+markers', name='Overnights visitors'))
fig_departure.add_trace(go.Scatter(x=df_dep_sameday.index, y=df_dep_sameday.values, mode='lines+markers', name='Same-day visitors'))

fig_departure.update_layout(xaxis_title='Year', yaxis_title='Number of departures')

# Pearson correlation (arrivals vs inbound expenditure)
correlation = df_arrivals.corr(df_inb_expend)

# AGE ANALYSIS
age = data.iloc[23:30, 1:].T

fig_age = go.Figure()
for age_group in age.columns:
    fig_age.add_trace(go.Bar(
        x=years,
        y=age[age_group],
        name=age_group
    ))

fig_age.update_layout(
    xaxis=dict(range=[2012, years[-1]]),
    xaxis_title='Year',
    yaxis_title='Population',
    barmode='stack'
)


# TOP DESTINATIONS
fig_top_dest =go.Figure()
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Croatia.values, mode='lines', name='Country name: Croatia'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Czech.values, mode='lines', name='Country name: Czech'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Germany.values, mode='lines', name='Country name: Germany'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Italy.values, mode='lines', name='Country name: Italy'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_UK.values, mode='lines', name='Country name: UK'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Greece.values, mode='lines', name='Country name: Greece'))

fig_top_dest.update_layout(xaxis=dict(range=[2014, years[-1]]), xaxis_title='Year', yaxis_title='Value (thousands)')

# REGRESSION MODEL
x = pd.concat([df_inb_expend, df_out_expend, df_departure], axis=1)  # independent variables
x = x.replace([np.inf, -np.inf], np.nan)  # replace inf with NaN
x = x.dropna()
y = df_arrivals.loc[x.index].values   # dependent variable

x_statsmodels = sm.add_constant(x) 
model_statsmodels = sm.OLS(y, x_statsmodels).fit()
predictions_statsmodels = model_statsmodels.predict(x_statsmodels)

mse_statsmodels = mean_squared_error(y, predictions_statsmodels)
r2_statsmodels = r2_score(y, predictions_statsmodels)

print("Statsmodels:")
print(model_statsmodels.summary())
print("Mean Squared Error (MSE):", mse_statsmodels)
print("R-squared:", r2_statsmodels)

line_x = np.array([x.iloc[:, 0].min(), x.iloc[:, 0].max()])
line_y = line_x * model_statsmodels.params[1] + model_statsmodels.params[0]


scatter_trace = go.Scatter(
    x=x.iloc[:, 0],
    y=y,
    mode='markers',
    name='Actual values'
)

regression_line_trace = go.Scatter(
    x=line_x,
    y=line_y,
    mode='lines',
    name='Regression line'
)

layout = go.Layout(
    xaxis={'title': 'Independent variable X'},
    yaxis={'title': 'Dependent variable Y'},
    showlegend=True
)
data5 = [scatter_trace, regression_line_trace]
figure_reg = go.Figure(data=data5, layout=layout)

# Binarization for F1 and accuracy calculation
binarizer = Binarizer(threshold=0.5)
y_binary_statsmodels = binarizer.transform(np.array(predictions_statsmodels).reshape(-1, 1)).flatten()
f1_statsmodels = f1_score(y_binary_statsmodels, binarizer.transform(y.reshape(-1, 1)).flatten(), average='weighted')
accuracy_statsmodels = accuracy_score(y_binary_statsmodels, binarizer.transform(y.reshape(-1, 1)).flatten())

print("Statsmodels:")
print(model_statsmodels.summary())
print("F1-score:", f1_statsmodels)
print("Accuracy:", accuracy_statsmodels)

x_statsmodels = sm.add_constant(x) 
model_statsmodels = sm.OLS(y, x_statsmodels).fit()
predictions_statsmodels = model_statsmodels.predict(x_statsmodels)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# SCikit-learn linear regression
model_sklearn = LinearRegression()
model_sklearn.fit(x_train, y_train)
predictions_sklearn = model_sklearn.predict(x_test)

y_binary_sklearn = binarizer.transform(predictions_sklearn.reshape(-1, 1)).flatten()
f1_sklearn = f1_score(y_test, y_binary_sklearn, average='weighted')
accuracy_sklearn = accuracy_score(y_test, y_binary_sklearn)

print("Scikit-learn:")
print("Regression coefficients:", model_sklearn.coef_)
print("F1-score:", f1_sklearn)
print("Accuracy:", accuracy_sklearn)


# Add index column for table display in dashboard
data['Index'] = data.index
data = data[['Index'] + list(data.columns[:-1])] 

app = dash.Dash(__name__) 

app.layout = html.Div([
    html.H1('Tourism in Poland'),
    dash_table.DataTable(data = data.to_dict('records'), columns=[{'name': col, 'id': col} for col in data.columns], page_size=10),

    html.H1('Tourist Arrivals Trend in Poland'),
    dcc.Graph(figure=fig_combined),
    html.Div('The highest trend in arrivals was before 2000, followed by a large drop. Since 2009, a slight increase can be observed. The drop in 2020 is due to Covid.'),
    html.Div('Most tourists arriving in Poland are same-day visitors. Far fewer people come for longer stays.'),


    html.H1('Tourist Departures Trend from Poland'),
    dcc.Graph(figure=fig_departure),

    html.H1('Percentage Change'),
    dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': "Both values", 'value': 'both'},
            {'label': 'Percentage change in arrivals', 'value': 'arrivals'},
            {'label': 'Percentage change in departures', 'value': 'departures'}
        ],
        value='both'
    ),
    dcc.Graph(id='percentage-change'),

    html.H1('Tourism Expenditure in Poland (in millions)'),
    dcc.Dropdown(
        id='data-dropdown2',
        options=[
            {'label': 'Total tourism expenditure in Poland', 'value': 'both2'},
            {'label': 'Travel-related expenditure in Poland', 'value': 'travel'},
            {'label': 'Transport-related expenditure in Poland', 'value': 'transport'}
        ],
        value='both2'
    ),
    dcc.Graph(id='inbound-expenditure'),

    html.H1('Poles’ Tourism Expenditure Abroad (in millions)'),
    dcc.Dropdown(
        id='data-dropdown3',
        options=[
            {'label': 'Total tourism expenditure abroad', 'value': 'both3'},
            {'label': 'Travel-related expenditure abroad', 'value': 'travel2'},
            {'label': 'Transport-related expenditure abroad', 'value': 'transport2'}
        ],
        value='both3'
    ),
    dcc.Graph(id='outbound-expenditure'),

    html.H1('Correlation between Arrivals and Expenditure'),
    dcc.Graph(
        id='corr-graph',
        figure={
            'data':[go.Scatter(x=df_arrivals, y=df_inb_expend, mode='markers', marker=dict(color='blue'))],
            'layout':go.Layout(xaxis={'title': 'Tourist arrivals'}, yaxis={'title': 'Expenditure'}, title=f'Correlation coefficient: {correlation: .2f}')
        }
    ),

    html.H1('Age Analysis of Travelers'),
    dcc.Graph(figure=fig_age),

    html.H1("Most Visited Countries by Poles"),
    dcc.Graph(figure=fig_top_dest),

    html.H1("Linear Regression Model"),
    html.H2("R-squared:"),
    html.Div(f"{r2_statsmodels}"),
    html.Div("The model explains approximately 60.47% of the variance in the dependent variable (number of arrivals in Poland). It is relatively well-fitted to the data."),
    html.H2("Mean Squared Error (MSE):"),
    html.Div(f"{mse_statsmodels}"),
    html.Div("A high MSE indicates that predicted values differ on average by this amount from actual values."),

    html.H1('Regression Plot'),
    dcc.Graph(
        id='regression-plot',
        figure=figure_reg
    ) 

])

@app.callback(
    Output('percentage-change', 'figure'),
    Input('data-dropdown', 'value')
)

def update_graph(selected_data):
    fig = go.Figure()
    
    if selected_data == 'both':
        fig.add_trace(go.Bar(x=df_arrivals.index, y=rate_arrivals,
                                 name='Percentage change in arrivals'))
        
        fig.add_trace(go.Bar(x=df_departure.index, y=rate_departure,
                             name='Percentage change in departures'))
    elif selected_data == 'arrivals':
        fig.add_trace(go.Bar(x=df_arrivals.index, y=rate_arrivals,
                                 name='Percentage change in arrivals'))
    elif selected_data == 'departures':
        fig.add_trace(go.Bar(x=df_departure.index, y=rate_departure,
                             name='Percentage change in departures'))
    
    fig.update_layout(xaxis_title='Year', yaxis_title='Percentage Change')
    
    return fig

@app.callback(
    Output('inbound-expenditure', 'figure'),
    Input('data-dropdown2', 'value')
)

def update_graph_inb_expend(selected_data):
    fig = go.Figure()

    if selected_data == 'both2':
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_inb_expend,
                                 name='Total tourism expenditure in Poland', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_travel_inb,
                                 name='Travel-related expenditure in Poland', marker_color='blue'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_transport_inb,
                                 name='Transport-related expenditure in Poland', marker_color='pink'))
    elif selected_data == 'travel':
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_inb_expend,
                                 name='Total tourism expenditure in Poland', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_travel_inb,
                                 name='Travel-related expenditure in Poland', marker_color='blue'))
    elif selected_data == 'transport':
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_inb_expend,
                                 name='Total tourism expenditure in Poland', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_transport_inb,
                                 name='Transport-related expenditure in Poland', marker_color='pink'))
    
    fig.update_layout(xaxis_title='Year', yaxis_title='Tourism expenditure in Poland (millions)',
                      barmode='stack')

    return fig


@app.callback(
    Output('outbound-expenditure', 'figure'),
    Input('data-dropdown3', 'value')
)

def update_graph_out_expend(selected_data):
    fig = go.Figure()

    if selected_data == 'both3':
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_out_expend,
                                 name='Total tourism expenditure abroad', marker_color='yellow'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_travel_out,
                                 name='Travel-related expenditure abroad', marker_color='orange'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_transport_out,
                                 name='Transport-related expenditure abroad', marker_color='green'))
    elif selected_data == 'travel2':
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_out_expend,
                                 name='Total tourism expenditure abroad', marker_color='yellow'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_travel_out,
                                 name='Travel-related expenditure abroad', marker_color='orange'))
    elif selected_data == 'transport2':
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_out_expend,
                                 name='Total tourism expenditure abroad', marker_color='yellow'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_transport_out,
                                 name='Transport-related expenditure abroad', marker_color='green'))
    
    fig.update_layout(xaxis_title='Year', yaxis_title='Tourism expenditure abroad (millions)',
                      barmode='stack')

    return fig

if __name__ == '__main__':
     app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
