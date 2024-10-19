import dash
from dash import dcc, html
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Load and prepare the dataset
df = pd.read_csv('The Rise Of Artificial Intellegence2.csv')

# Ensure 'Year' column is correctly formatted as integers
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)

# Ensure relevant columns are numeric
df['AI Adoption (%)'] = df['AI Adoption (%)'].astype(float)
df['AI Software Revenue(in Billions)'] = df['AI Software Revenue(in Billions)'].astype(float)
df['Estimated Jobs Eliminated by AI (millions)'] = df['Estimated Jobs Eliminated by AI (millions)'].astype(float)
df['Estimated New Jobs Created by AI (millions)'] = df['Estimated New Jobs Created by AI (millions)'].astype(float)

# Compute the correlation matrix
correlation_matrix = df[['AI Adoption (%)', 'Estimated Jobs Eliminated by AI (millions)', 'Estimated New Jobs Created by AI (millions)']].corr()

# Extract the correlations of interest
correlation_adoption_jobs_eliminated = correlation_matrix.loc['AI Adoption (%)', 'Estimated Jobs Eliminated by AI (millions)']
correlation_adoption_jobs_created = correlation_matrix.loc['AI Adoption (%)', 'Estimated New Jobs Created by AI (millions)']

# Prepare the data for Linear Regression
X = df[['Year']]  # Independent variable (Year)
y_adoption = df['AI Adoption (%)']  # Dependent variable for AI adoption
y_revenue = df['AI Software Revenue(in Billions)']  # Dependent variable for Revenue growth
y_jobs_eliminated = df['Estimated Jobs Eliminated by AI (millions)']  # Dependent variable for jobs eliminated
y_jobs_created = df['Estimated New Jobs Created by AI (millions)']  # Dependent variable for jobs created

# Initialize and fit the Linear Regression models
model_adoption = LinearRegression()
model_adoption.fit(X, y_adoption)

model_revenue = LinearRegression()
model_revenue.fit(X, y_revenue)

model_jobs_eliminated = LinearRegression()
model_jobs_eliminated.fit(X, y_jobs_eliminated)

model_jobs_created = LinearRegression()
model_jobs_created.fit(X, y_jobs_created)
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load and prepare the dataset
df = pd.read_csv('The Rise Of Artificial Intellegence2.csv')

# Ensure 'Year' column is correctly formatted as integers
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)

# Ensure relevant columns are numeric
df['AI Adoption (%)'] = df['AI Adoption (%)'].astype(float)
df['AI Software Revenue(in Billions)'] = df['AI Software Revenue(in Billions)'].astype(float)
df['Estimated Jobs Eliminated by AI (millions)'] = df['Estimated Jobs Eliminated by AI (millions)'].astype(float)
df['Estimated New Jobs Created by AI (millions)'] = df['Estimated New Jobs Created by AI (millions)'].astype(float)

# Compute the correlation matrix
correlation_matrix = df[['AI Adoption (%)', 'Estimated Jobs Eliminated by AI (millions)', 'Estimated New Jobs Created by AI (millions)']].corr()

# Extract the correlations of interest
correlation_adoption_jobs_eliminated = correlation_matrix.loc['AI Adoption (%)', 'Estimated Jobs Eliminated by AI (millions)']
correlation_adoption_jobs_created = correlation_matrix.loc['AI Adoption (%)', 'Estimated New Jobs Created by AI (millions)']

# Prepare the data for Linear Regression
X = df[['Year']]  # Independent variable (Year)
y_adoption = df['AI Adoption (%)']  # Dependent variable for AI adoption
y_revenue = df['AI Software Revenue(in Billions)']  # Dependent variable for Revenue growth
y_jobs_eliminated = df['Estimated Jobs Eliminated by AI (millions)']  # Dependent variable for jobs eliminated
y_jobs_created = df['Estimated New Jobs Created by AI (millions)']  # Dependent variable for jobs created

# Initialize and fit the Linear Regression models
model_adoption = LinearRegression()
model_adoption.fit(X, y_adoption)

model_revenue = LinearRegression()
model_revenue.fit(X, y_revenue)

model_jobs_eliminated = LinearRegression()
model_jobs_eliminated.fit(X, y_jobs_eliminated)

model_jobs_created = LinearRegression()
model_jobs_created.fit(X, y_jobs_created)

# Predict future values
future_years = pd.DataFrame({'Year': [2025, 2026]})
adoption_predictions = model_adoption.predict(future_years)
revenue_predictions = model_revenue.predict(future_years)
jobs_eliminated_predictions = model_jobs_eliminated.predict(future_years)
jobs_created_predictions = model_jobs_created.predict(future_years)

# Calculate year-over-year change for AI adoption
df['AI Adoption YoY Change (%)'] = df['AI Adoption (%)'].pct_change() * 100

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AI Trends Dashboard"),
    
    html.Div([
        html.H3("Correlation between AI Adoption and Jobs Eliminated:"),
        html.P(f"{correlation_adoption_jobs_eliminated:.2f}"),
        html.H3("Correlation between AI Adoption and Jobs Created:"),
        html.P(f"{correlation_adoption_jobs_created:.2f}")
    ]),
    
    html.Div([
        html.H3("Year-over-Year Change in AI Adoption (%):"),
        dcc.Graph(
            id='ai-adoption-yoy',
            figure=px.line(df, x='Year', y='AI Adoption YoY Change (%)', title='Year-over-Year Change in AI Adoption (%)')
        )
    ]),
    
    dcc.Graph(
        id='ai-adoption',
        figure=px.scatter(df, x='Year', y='AI Adoption (%)', title='AI Adoption Over Time')
    ),
    
    dcc.Graph(
        id='ai-revenue',
        figure=px.scatter(df, x='Year', y='AI Software Revenue(in Billions)', title='AI Software Revenue Growth Over Time')
    ),
    
    dcc.Graph(
        id='jobs',
        figure=go.Figure([
            go.Scatter(x=df['Year'], y=df['Estimated Jobs Eliminated by AI (millions)'], mode='lines+markers', name='Jobs Eliminated'),
            go.Scatter(x=df['Year'], y=df['Estimated New Jobs Created by AI (millions)'], mode='lines+markers', name='Jobs Created'),
            go.Scatter(x=future_years['Year'], y=jobs_eliminated_predictions, mode='lines+markers', name='Predicted Jobs Eliminated', line=dict(dash='dash')),
            go.Scatter(x=future_years['Year'], y=jobs_created_predictions, mode='lines+markers', name='Predicted Jobs Created', line=dict(dash='dash'))
        ]).update_layout(title='Jobs Eliminated vs New Jobs Created by AI', xaxis_title='Year', yaxis_title='Jobs (millions)')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
# Calculate year-over-year change
df['AI Adoption YoY Change (%)'] = df['AI Adoption (%)'].pct_change() * 100
df['AI Software Revenue YoY Change (%)'] = df['AI Software Revenue(in Billions)'].pct_change() * 100

# Print the year-over-year changes
print("Year-over-Year Change in AI Adoption (%):")
print(df[['Year', 'AI Adoption YoY Change (%)']])

print("\nYear-over-Year Change in AI Software Revenue (%):")
print(df[['Year', 'AI Software Revenue YoY Change (%)']])

# Predict future values
future_years = pd.DataFrame({'Year': [2025, 2026]})
adoption_predictions = model_adoption.predict(future_years)
revenue_predictions = model_revenue.predict(future_years)
jobs_eliminated_predictions = model_jobs_eliminated.predict(future_years)
jobs_created_predictions = model_jobs_created.predict(future_years)

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AI Trends Dashboard"),
    
    html.Div([
        html.H3("Correlation between AI Adoption and Jobs Eliminated:"),
        html.P(f"{correlation_adoption_jobs_eliminated:.2f}"),
        html.H3("Correlation between AI Adoption and Jobs Created:"),
        html.P(f"{correlation_adoption_jobs_created:.2f}")
    ]),
    
    dcc.Graph(
        id='ai-adoption',
        figure=px.scatter(df, x='Year', y='AI Adoption (%)', title='AI Adoption Over Time')
    ),
    
    dcc.Graph(
        id='ai-revenue',
        figure=px.scatter(df, x='Year', y='AI Software Revenue(in Billions)', title='AI Software Revenue Growth Over Time')
    ),
    
    dcc.Graph(
        id='jobs',
        figure=go.Figure([
            go.Scatter(x=df['Year'], y=df['Estimated Jobs Eliminated by AI (millions)'], mode='lines+markers', name='Jobs Eliminated'),
            go.Scatter(x=df['Year'], y=df['Estimated New Jobs Created by AI (millions)'], mode='lines+markers', name='Jobs Created'),
            go.Scatter(x=future_years['Year'], y=jobs_eliminated_predictions, mode='lines+markers', name='Predicted Jobs Eliminated', line=dict(dash='dash')),
            go.Scatter(x=future_years['Year'], y=jobs_created_predictions, mode='lines+markers', name='Predicted Jobs Created', line=dict(dash='dash'))
        ]).update_layout(title='Jobs Eliminated vs New Jobs Created by AI', xaxis_title='Year', yaxis_title='Jobs (millions)')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
