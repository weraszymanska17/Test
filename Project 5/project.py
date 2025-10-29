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




data = pd.read_csv("/Users/weronika/Desktop/studia/informatyka/2 semestr/PAD/PROJEKT/dane.csv", delimiter=';', index_col=0)
# print(data)

# zamiana spacji na puste miejsca
data = data.replace('\s', '', regex=True)

# zamiana NaN na puste miejsca
data = data.fillna('')

# usuwanie spacji z wartości w kolumnach
data = data.apply(lambda x: x.str.replace(' ', ''))

# zamiana przecinków na kropki
data = data.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)

# zamiana na wartosci numeryczne
data = data.apply(pd.to_numeric)


df_arrivals = data.iloc[0] # wiersze dotyczące przyjazdów
df_arrivals = df_arrivals.iloc[1:] # kolumny z latami
df_arrivals = df_arrivals.apply(pd.to_numeric)

df_overnights = data.iloc[1]
df_overnights = df_overnights.iloc[1:]
df_overnights = df_overnights.apply(pd.to_numeric)

df_sameday = data.iloc[2]
df_sameday = df_sameday.iloc[1:]
df_sameday = df_sameday.apply(pd.to_numeric)

df_departure = data.iloc[16, 1:].apply(pd.to_numeric)
df_dep_overnights = data.iloc[17, 1:].apply(pd.to_numeric)
df_dep_sameday = data.iloc[18, 1:].apply(pd.to_numeric)

df_inb_expend = data.iloc[3, 1:].apply(pd.to_numeric)
df_travel_inb = data.iloc[4, 1:].apply(pd.to_numeric)
df_transport_inb = data.iloc[5, 1:].apply(pd.to_numeric)

df_out_expend = data.iloc[19, 1:].apply(pd.to_numeric)
df_travel_out = data.iloc[20, 1:].apply(pd.to_numeric)
df_transport_out = data.iloc[21, 1:].apply(pd.to_numeric)

df_Croatia = data.iloc[10, 1:].apply(pd.to_numeric)
df_Czech  = data.iloc[11, 1:].apply(pd.to_numeric)
df_Germany = data.iloc[12, 1:].apply(pd.to_numeric)
df_Italy = data.iloc[13, 1:].apply(pd.to_numeric)
df_UK = data.iloc[14, 1:].apply(pd.to_numeric)
df_Greece = data.iloc[15, 1:].apply(pd.to_numeric)

years = data.columns[1:].astype(int)

# WYKRES PRZYJAZDY
fig_combined = go.Figure()

fig_combined.add_trace(go.Scatter(x=df_arrivals.index, y=df_arrivals.values, mode='lines+markers', name='Total arrivals'))
fig_combined.add_trace(go.Scatter(x=df_overnights.index, y=df_overnights.values, mode='lines+markers', name='Overnights visitors'))
fig_combined.add_trace(go.Scatter(x=df_sameday.index, y=df_sameday.values, mode='lines+markers', name='Same-day visitors'))

fig_combined.update_layout(xaxis_title='Rok', yaxis_title='Liczba przyjazdów')

# wzrost/spadek liczby przyjazdów
rate_arrivals = df_arrivals.pct_change()*100
rate_arrivals = rate_arrivals[1:] #dla pierwszego brak danych 

rate_departure = df_departure.pct_change()*100
rate_departure = rate_departure[1:] #dla pierwszego brak danych 


# WYKRES WYJAZDY
fig_departure = go.Figure()

fig_departure.add_trace(go.Scatter(x=df_departure.index, y=df_departure.values, mode='lines+markers', name='Total departure'))
fig_departure.add_trace(go.Scatter(x=df_dep_overnights.index, y=df_dep_overnights.values, mode='lines+markers', name='Overnights visitors'))
fig_departure.add_trace(go.Scatter(x=df_dep_sameday.index, y=df_dep_sameday.values, mode='lines+markers', name='Same-day visitors'))

fig_departure.update_layout(xaxis_title='Rok', yaxis_title='Liczba wyjazdów')

# współczynnik korelacji Pearsona (przyjazdy a wydatki krajowe)
correlation = df_arrivals.corr(df_inb_expend)

# analiza wieku
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
    xaxis_title='Rok',
    yaxis_title='Liczba ludności',
    barmode='stack'
)


# kraje do których jezdzą Polacy
fig_top_dest =go.Figure()
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Croatia.values, mode='lines', name='Country name: Croatia'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Czech.values, mode='lines', name='Country name: Czech'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Germany.values, mode='lines', name='Country name: Germany'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Italy.values, mode='lines', name='Country name: Italy'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_UK.values, mode='lines', name='Country name: UK'))
fig_top_dest.add_trace(go.Scatter(x=years, y=df_Greece.values, mode='lines', name='Country name: Greece'))

fig_top_dest.update_layout(xaxis=dict(range=[2014, years[-1]]), xaxis_title='Rok', yaxis_title='Wartość (w tys.)')

# model regresji
x = pd.concat([df_inb_expend, df_out_expend, df_departure], axis=1)  # zmienne niezależne
x = x.replace([np.inf, -np.inf], np.nan)  # zamiana wartości inf na NaN
x = x.dropna()
y = df_arrivals.loc[x.index].values  # zmienna zalezna 

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
    name='Wartości rzeczywiste'
)

regression_line_trace = go.Scatter(
    x=line_x,
    y=line_y,
    mode='lines',
    name='Linia regresji'
)

layout = go.Layout(
    xaxis={'title': 'Zmienna niezalezna X'},
    yaxis={'title': 'Zmienna zalezna Y'},
    showlegend=True
)
data5 = [scatter_trace, regression_line_trace]
figure_reg = go.Figure(data=data5, layout=layout)

binarizer = Binarizer(threshold=0.5)
y_binary_statsmodels = binarizer.transform(np.array(predictions_statsmodels).reshape(-1, 1)).flatten()
f1_statsmodels = f1_score(y_binary_statsmodels, binarizer.transform(y.reshape(-1, 1)).flatten(), average='weighted')
accuracy_statsmodels = accuracy_score(y_binary_statsmodels, binarizer.transform(y.reshape(-1, 1)).flatten())

print("Statsmodels:")
print(model_statsmodels.summary())
print("F1-score:", f1_statsmodels)
print("Accuracy:", accuracy_statsmodels)

# x = sm.add_constant(x) 
x_statsmodels = sm.add_constant(x) 
model_statsmodels = sm.OLS(y, x_statsmodels).fit()
predictions_statsmodels = model_statsmodels.predict(x_statsmodels)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model regresji liniowej oparty na bibliotece scikit-learn
model_sklearn = LinearRegression()
model_sklearn.fit(x_train, y_train)
predictions_sklearn = model_sklearn.predict(x_test)

# Obliczanie F1-score i dokładności dla modelu scikit-learn
y_binary_sklearn = binarizer.transform(predictions_sklearn.reshape(-1, 1)).flatten()
f1_sklearn = f1_score(y_test, y_binary_sklearn, average='weighted')
accuracy_sklearn = accuracy_score(y_test, y_binary_sklearn)

print("Scikit-learn:")
print("Współczynniki regresji:", model_sklearn.coef_)
print("F1-score:", f1_sklearn)
print("Dokładność (accuracy):", accuracy_sklearn)


# dodaje kolumne index (pierwszą), zeby było ją widać w tablicy na dashboardzie
data['Index'] = data.index
data = data[['Index'] + list(data.columns[:-1])] 

app = dash.Dash(__name__) 

app.layout = html.Div([
    html.H1('Turystyka w Polsce'),
    dash_table.DataTable(data = data.to_dict('records'), columns=[{'name': col, 'id': col} for col in data.columns], page_size=10),

    html.H1('Trend przyjazdów turystycznych do Polski'),
    dcc.Graph(figure=fig_combined),
    html.Div('Największy trend przyjazdów do Polski występował przed rokiem 2000, później obserwujemy ogromny spadek. Od roku 2009 delikatny wzrost i linia trendu kształtuje się w miarę stabilnie. Spadek w 2020 roku wynika z Covid.'),
    html.Div('Co więcej mozna zaobserwować, ze większość turystów przyjezdzających do Polski to turyści jednodniowi. Zdecydowanie mniej osób przyjezdza na pobyty dłuzsze.'),

    html.H1('Trend wyjazdów turystycznych z Polski'),
    dcc.Graph(figure=fig_departure),

    html.H1('Procentowa zmiana'),
    dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': "Obydwie wartości", 'value': 'both'},
            {'label': 'Procentowy wzrost/spadek przyjazdów', 'value': 'arrivals'},
            {'label': 'Procentowy wzrost/spadek wyjazdów', 'value': 'departures'}
        ],
        value='both'
    ),
    dcc.Graph(id='procentowa_zmiana'),

    html.H1('Analiza wydatków (w milionach) turystycznych w Polsce'),
    dcc.Dropdown(
        id='data-dropdown2',
        options=[
            {'label': 'Łączne wydatki związane z turystyką w Polsce', 'value': 'both2'},
            {'label': 'Wydatki związane z podrózami w Polsce', 'value': 'travel'},
            {'label': 'Wydatki związane z trasportem w Polsce', 'value': 'transport'}
        ],
        value='both2'
    ),
    dcc.Graph(id='inbound-expenditure'),

    html.H1('Analiza wydatków (w milionach) turystycznych Polaków za granicą'),
    dcc.Dropdown(
        id='data-dropdown3',
        options=[
            {'label': 'Łączne wydatki turystyczne Polaków za granicą', 'value': 'both3'},
            {'label': 'Wydatki związane z podrózami Polaków za granicą', 'value': 'travel2'},
            {'label': 'Wydatki związane z trasportem Polaków za granicą', 'value': 'transport2'}
        ],
        value='both3'
    ),
    dcc.Graph(id='outbound-expenditure'),

    html.H1('Korelacja między przyjazdami turystycznymi a wydatkami'),
    dcc.Graph(
        id='corr-graph',
        figure={
            'data':[go.Scatter(x=df_arrivals, y=df_inb_expend, mode='markers', marker=dict(color='blue'))],
            'layout':go.Layout(xaxis={'title': 'Przyjazdy turystyczne'}, yaxis={'title': 'Wydatki'}, title=f'Współczynnik korelacji: {correlation: .2f}')
        }
    ),

    html.H1('Analiza wieku osób podrózujących'),
    dcc.Graph(figure=fig_age),

    html.H1("Kraje, do których Polacy wyjezdzają najczęściej"),
    dcc.Graph(figure=fig_top_dest),

    html.H1("Model regresji liniowej"),
    # html.H2("Accuracy:"),
    # html.Div(f"{accuracy}"),
    # html.H2("F1 Score:"),
    # html.Div(f"{f1}"),
    html.H2("R-squared:"),
    html.Div(f"{r2_statsmodels}"),
    html.Div("Model wyjaśnia około 60,47% zmienności zmiennej zaleznej (liczby przyjazdów do Polski). Model ma relatywnie wysoką wartość R^2, mona uznać, ze model dobrze dopasowuje się do danych. "),
    html.H2("Mean Squared Error (MSE):"),
    html.Div(f"{mse_statsmodels}"),
    html.Div("Średni błąd kwadratów jest wysoki, co oznacza, ze przewidywane wartości zmiennej zaleznej róznią się średnio o wartość MSE od wartości rzeczywistych."),

    html.H1('Wykres regresji'),
    dcc.Graph(
        id='regression-plot',
        figure=figure_reg
    ) 

])

@app.callback(
    Output('procentowa_zmiana', 'figure'),
    Input('data-dropdown', 'value')
)

def update_graph(selected_data):
    fig = go.Figure()
    
    if selected_data == 'both':
        fig.add_trace(go.Bar(x=df_arrivals.index, y=rate_arrivals,
                                 name='Procentowy wzrost/spadek przyjazdów'))
        
        fig.add_trace(go.Bar(x=df_departure.index, y=rate_departure,
                             name='Procentowy wzrost/spadek wyjazdów'))
    elif selected_data == 'arrivals':
        fig.add_trace(go.Bar(x=df_arrivals.index, y=rate_arrivals,
                                 name='Procentowy wzrost/spadek przyjazdów'))
    elif selected_data == 'departures':
        fig.add_trace(go.Bar(x=df_departure.index, y=rate_departure,
                             name='Procentowy wzrost/spadek wyjazdów'))
    
    fig.update_layout(xaxis_title='Rok', yaxis_title='Procentowa zmiana')
    
    return fig

@app.callback(
    Output('inbound-expenditure', 'figure'),
    Input('data-dropdown2', 'value')
)

def update_graph_inb_expend(selected_data):
    fig = go.Figure()

    if selected_data == 'both2':
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_inb_expend,
                                 name='Łączne wydatki związane z turystyką w Polsce', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_travel_inb,
                                 name='Wydatki związane z podrózami w Polsce', marker_color='blue'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_transport_inb,
                                 name='Wydatki związane z trasportem w Polsce', marker_color='pink'))
    elif selected_data == 'travel':
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_inb_expend,
                                 name='Łączne wydatki związane z turystyką w Polsce', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_travel_inb,
                                 name='Wydatki związane z podrózami w Polsce', marker_color='blue'))
    elif selected_data == 'transport':
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_inb_expend,
                                 name='Łączne wydatki związane z turystyką w Polsce', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_inb_expend.index, y=df_transport_inb,
                                 name='Wydatki związane z trasportem w Polsce', marker_color='pink'))
    
    fig.update_layout(xaxis_title='Rok', yaxis_title='Wydatki turystyczne w Polsce (w milionach)',
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
                                 name='Łączne wydatki turystyczne Polaków za granicą', marker_color='yellow'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_travel_out,
                                 name='Wydatki związane z podrózami Polaków za granicą', marker_color='orange'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_transport_out,
                                 name='Wydatki związane z trasportem Polaków za granicą', marker_color='green'))
    elif selected_data == 'travel2':
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_out_expend,
                                 name='Łączne wydatki turystyczne Polaków za granicą', marker_color='yellow'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_travel_out,
                                 name='Wydatki związane z podrózami Polaków za granicą', marker_color='orange'))
    elif selected_data == 'transport2':
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_out_expend,
                                 name='Łączne wydatki turystyczne Polaków za granicą', marker_color='yellow'))
        fig.add_trace(go.Bar(x=df_out_expend.index, y=df_transport_out,
                                 name='Wydatki związane z trasportem Polaków za granicą', marker_color='green'))
    
    fig.update_layout(xaxis_title='Rok', yaxis_title='Wydatki turystyczne Polaków za granicą (w milionach)',
                      barmode='stack')

    return fig

if __name__ == '__main__':
     app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))