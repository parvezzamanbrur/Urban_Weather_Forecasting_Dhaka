import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import cufflinks as cf
import datetime as dt
import math

import plotly
import plotly.io as pio
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.offline as py

#import sklearn
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#from sklearn.metrics import mean_absolute_percentage_error
#from sklearn.preprocessing import RobustScaler, MinMaxScaler

##import statsmodels
import statsmodels.api as sm
#from statsmodels.tsa.stattools import adfuller, kpss
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX

#from pmdarima import auto_arima

#from prophet import Prophet
#from prophet.plot import plot_plotly

#import tensorflow as tf
#from tf.keras.models import Sequential
#from tf.keras.layers import SimpleRNN, Dense
#from tf.keras.callbacks import EarlyStopping
#from tf.keras.layers import LSTM, Bidirectional
#from tf.keras.layers import GRU

plt.rcParams["figure.figsize"] = (10,6)
sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
df = pd.read_csv("Data/dhakaweathertrain.csv",
                 parse_dates=['date'],  # change to date time format
                 index_col="date")
#df = df.drop("dewpoint", axis=1)
df = df.rename(columns={'temp': 'meantemp'})
print(df)
print(df.head())
missing_val = df.isin(['?', '', 'None', 'NaN']).sum()
print(missing_val)
df.info()
print(df.describe().T)
# creating new features for EDA
df["year"] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df["date"] = df.index.date
print(df.head())
df.info()
# creating dark gray template(xgridoff_dark) from xgridoff template:
# Get the 'xgridoff' template
xgridoff_template = pio.templates['xgridoff']
# Customize the template for dark mode
xgridoff_template.layout.paper_bgcolor = 'rgb(25, 25, 25)'  # very dark gray background
xgridoff_template.layout.plot_bgcolor = 'rgb(35, 35, 35)'  # Dark gray plot background
xgridoff_template.layout.font.color = 'lightgray'  # Light gray font color
# Adjust gridline color and width
xgridoff_template.layout.xaxis.gridcolor = 'rgba(200, 200, 200, 0.3)'  # Light gray with transparency
xgridoff_template.layout.yaxis.gridcolor = 'rgba(200, 200, 200, 0.3)'  # Light gray with transparency
xgridoff_template.layout.xaxis.gridwidth = 1  # Set gridline width
xgridoff_template.layout.yaxis.gridwidth = 1  # Set gridline width
# Update Plotly templates with the modified 'xgridoff' template
pio.templates['xgridoff_dark'] = xgridoff_template
fig_meantemp = px.line(df, x=df.index, y='meantemp', title='Mean Temperature Over Time')
fig_meantemp.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
fig_meantemp.write_html("Mean Temperature Over Time.html")
fig_humidity = px.line(df, x=df.index, y='humidity', title='Humidity Over Time')
fig_humidity.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
fig_humidity.write_html("Humidity Over Time.html")
fig_wind_speed = px.line(df, x=df.index, y='wind_speed', title='Wind Speed Over Time')
fig_wind_speed.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
fig_wind_speed.write_html("Wind Speed Over Time.html")
fig_meanpressure = px.line(df, x=df.index, y='meanpressure', title='Mean Pressure Over Time')
fig_meanpressure.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
fig_meanpressure.write_html("Mean presuure Over Time.html")
fig_meanrainfall = px.line(df, x=df.index, y='meanrainfall', title='Mean Rainfall Over Time')
fig_meanrainfall.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
fig_meanrainfall.write_html("Mean Rainfall Over Time.html")
fig_dewpoint = px.line(df, x=df.index, y='meanpressure', title='Mean Dewpoint Over Time')
fig_dewpoint.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
fig_dewpoint.write_html("Mean Dewpoint Over Time.html")
# Perform seasonal decomposition
result = seasonal_decompose(df['meantemp'], model='additive', period=365)
# Plot the decomposed components
fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                       subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
fig.add_trace(go.Scatter(x=df.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)
fig.update_layout(template= 'xgridoff_dark',height=800, title='Seasonal Decomposition of Mean Temperature')
#fig.show(renderer="iframe") # Plot Render Issue Solve
fig.write_html("Seasonal Decompositions.html") ## Plot Render Issue Solve
plt.rcParams['figure.figsize'] = [15, 7]
# Select the 'meantemp' column and resample it to monthly frequency
data_monthly = df['meantemp'].resample('ME').mean()
# Perform seasonal decomposition for 'meantemp' feature
decomposition = sm.tsa.seasonal_decompose(data_monthly)
# Plot the decomposition
fig = decomposition.plot()
plt.show()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 8))
sns.set_style('darkgrid')
# Plot each parameter against the 'month'
sns.lineplot(x='month', y='meantemp', data=df, color='purple', ax=axes[0][0])
axes[0][0].set_title('Mean Temperature Over Month')
sns.lineplot(x='month', y='humidity', data=df, color='darkorange', ax=axes[1][0])
axes[1][0].set_title('Humidity Over Month')
sns.lineplot(x='month', y='meanpressure', data=df, color='green', ax=axes[0][1])
axes[0][1].set_title('Mean Pressure Over Month')
sns.lineplot(x='month', y='wind_speed', data=df, color='darkcyan', ax=axes[1][1])
axes[1][1].set_title('Wind Speed Over Month')
sns.lineplot(x='month', y='meanrainfall', data=df, color='darkcyan', ax=axes[2][0])
axes[2][0].set_title('Mean Rainfall Over Month')
sns.lineplot(x='month', y='dewpoint', data=df, color='darkcyan', ax=axes[2][1])
axes[2][1].set_title('Dewpoint Over Month')
plt.tight_layout()
plt.show()
# Create a figure with the specified size
plt.figure(figsize=(18, 6))
sns.set_style('darkgrid')
plt.title("Temperature Over the Years")
sns.lineplot(data=df, x='month', y='meantemp', hue='year', palette='plasma', errorbar=None);
plt.figure(figsize=(13, 6))
sns.heatmap(df.select_dtypes('number').corr(), cmap='Blues', annot=True, fmt='.2f')
plt.show()
plt.figure(figsize=(15, 8))
sns.set_style('darkgrid')
correlation_matrix = round(df.select_dtypes('number').corr(), 2)
correlation_with_trgt = correlation_matrix['meantemp'].sort_values(ascending=False)
ax = sns.barplot(x=correlation_with_trgt.index, y=correlation_with_trgt, palette='viridis')
plt.title('Correlation with meantemp', size= 20)
plt.xlabel('Features')
plt.ylabel('Correlation')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.xticks(rotation=45, ha='right')
plt.show()
# Increase the size of the figure
fig = px.scatter_matrix(df,
                        dimensions=['meantemp', 'humidity', 'wind_speed', 'meanpressure'],
                        title='Scatter Matrix of Weather Parameters')
# Update layout to increase figure size and add outlines to dots
fig.update_layout(
    width=1200,  # Width of the figure
    height=800,  # Height of the figure
    title_x=0.5,  # Title position
    margin=dict(l=50, r=50, t=50, b=50),  # Margin around the plot
    template='xgridoff_dark')
# Update marker properties to add outlines
fig.update_traces(marker=dict(line=dict(width=1, color='black')))  # Add outlines to dots
# Show the figure
fig.write_html("Scatter Plot.html")
# Reset seaborn style to default
sns.set_style('darkgrid')
# Create subplots
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
# Plot histograms with KDE
sns.histplot(data=df, x="meantemp", ax=ax[0][0], kde=True)
sns.histplot(data=df, x="humidity", ax=ax[0][1], kde=True, color='green')
sns.histplot(data=df, x="wind_speed", ax=ax[1][0], kde=True, color='slateblue')
sns.histplot(data=df, x="meanpressure", ax=ax[1][1], kde=True, color='teal')
sns.histplot(data=df, x="meanrainfall", ax=ax[2][0], kde=True, color='darkcyan')
sns.histplot(data=df, x="dewpoint", ax=ax[2][1], kde=True, color='blue')
# Show the plot
plt.show()
sns.set_style('darkgrid')
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(14, 10))
# List of parameters and their titles
parameters = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
titles = ['Monthly Average Mean Temperature', 'Monthly Average Humidity',
          'Monthly Average Wind Speed', 'Monthly Average Pressure' ]
for ax, param, title in zip(axs.flatten(), parameters, titles):
    # Grouping the data by month, calculating the average mean value for each month
    monthly_average = df.groupby('month')[param].mean()
    # Polar Plot theta (angle) and radii (length) settings
    theta = np.linspace(0, 2 * np.pi, len(monthly_average), endpoint=False)
    radii = monthly_average.values
    # Extend theta and radii to connect the circle
    theta = np.append(theta, theta[0])
    radii = np.append(radii, radii[0])
    # Polar Plot
    ax.plot(theta, radii)
    ax.set_title(title, va='bottom')
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May',
                        'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylim(0, radii.max() + 10)
plt.tight_layout()
plt.show()