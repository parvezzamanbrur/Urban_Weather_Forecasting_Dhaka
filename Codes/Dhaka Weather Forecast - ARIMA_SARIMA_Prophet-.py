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

import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
# Feather Selection
df = df[['meantemp', 'humidity', 'wind_speed', 'meanpressure', 'meanrainfall', 'dewpoint']]
df.head()
# Creating a new Dataframe for ARIMA-SARIMA Models
df1 = df.copy()
print(df1.head())
# Check Stationary
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Plot ACF
plot_acf(df1['meantemp'], ax=axes[0], lags=40, title='Autocorrelation Function (ACF) for Mean Temperature')
# Plot PACF
plot_pacf(df1['meantemp'], ax=axes[1], lags=40, title='Partial Autocorrelation Function (PACF) for Mean Temperature')
plt.show()
The Series does not seem stationary, ACF has to become 0 at some point.
But we are not certainly sure yet.
Let's confirm with the ADF and KPSS tests..
ADF and KPSS
def check_stationarity(series):
    print(f'\n___________________Checking Stationarity for: {series.name}___________________\n')
    # ADF Test
    adf_test = adfuller(series.values)
    print('ADF Test:\n')
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
        print('\t%s: %.3f' % (key, value))

    if (adf_test[1] <= 0.05) & (adf_test[4]['5%'] > adf_test[0]):
        print("\u001b[32mSeries is Stationary (ADF Test)\u001b[0m")
    else:
        print("\x1b[31mSeries is Non-stationary (ADF Test)\x1b[0m")
    print('\n' + '-' * 50 + '\n')
    # KPSS Test
    kpss_test = kpss(series.values, regression='c', nlags='auto')
    print('KPSS Test:\n')
    print('KPSS Statistic: %f' % kpss_test[0])
    print('p-value: %f' % kpss_test[1])
    print('Critical Values:')
    for key, value in kpss_test[3].items():
        print('\t%s: %.3f' % (key, value))
    if kpss_test[1] > 0.05:
        print("\u001b[32mSeries is Stationary (KPSS Test)\u001b[0m")
    else:
        print("\x1b[31mSeries is Non-stationary (KPSS Test)\x1b[0m")
# Check initial stationarity for each feature
check_stationarity(df1['meantemp'])
check_stationarity(df1['humidity'])
check_stationarity(df1['wind_speed'])
check_stationarity(df1['meanpressure'])
check_stationarity(df1['meanrainfall'])
check_stationarity(df1['dewpoint'])
Apply differencing to make the series stationary
df1['meantemp_diff'] = df1['meantemp'].diff().fillna(0)  # diff() default is 1st degree differencing
print('\n\n############################### After Differencing ###############################\n\n')
check_stationarity(df1['meantemp_diff'])
f, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
plot_acf(df1['meantemp'], lags=40, ax=ax[0, 0], title='ACF on non-stationary')
plot_pacf(df1['meantemp'], lags=40, ax=ax[0, 1], method='ols', title='PACF on non-stationary')
plot_acf(df1['meantemp_diff'], lags=40, ax=ax[1, 0], title='ACF on differenced/stationary')
plot_pacf(df1['meantemp_diff'], lags=40, ax=ax[1, 1], method='ols', title='PACF on differenced/stationary')
plt.tight_layout()
plt.show()
# Interpreting ACF and PACF
### https://spureconomics.com/interpreting-acf-and-pacf-plots/
# 1. Autocorrelation Function (ACF):
# The ACF measures the correlation between a time series and its lagged values at different lag intervals.
# On the x-axis, you have the lag or time interval.
# The y-axis represents the correlation coefficient, which ranges from -1 to 1.
# Positive values indicate positive correlation (similar movement in the same direction), while negative values indicate negative correlation (opposite movement).
# The shaded region represents the 95% confidence interval. Correlation values outside this region are considered statistically significant.
# Significant spikes in the ACF plot indicate potential seasonal patterns or trends in the data.
# 2. Partial Autocorrelation Function (PACF):
# The PACF measures the correlation between a time series and its lagged values, controlling for the effects of other lags in between.
# It helps identify the direct relationship between a data point and its lagged values, excluding the influence of other lags.
# Similar to the ACF plot, the x-axis represents the lag or time interval, and the y-axis represents the correlation coefficient.
# Significant spikes in the PACF plot indicate the number of lag terms to include in autoregressive models (AR).
# Interpreting these plots helps identify potential patterns, trends, and seasonality in the time series data, providing insights for modeling and forecasting.
# If there are clear patterns like seasonality in the ACF and PACF plots, it suggests that the data exhibits periodic behavior.
# This implies that there are recurring patterns or cycles in the data over time, which can be important for understanding and forecasting future values.

# Split the data into training and testing sets
train_size = int(len(df1) * 0.8)
train, test = df1.iloc[:train_size], df1.iloc[train_size:]
print(f'Train shape: {train.shape}')
print(train.head())
print(f'Test shape: {test.shape}')
print(test.head())
print(df1.info())
# Fit the ARIMA model
arima_model = ARIMA(train['meantemp'], order=(2,1,2))  # ARIMA(p,d,q)
arima_model_fit = arima_model.fit()
print(arima_model_fit.summary())
# Make predictions
arima_pred = arima_model_fit.forecast(steps=len(test))
# Calculate error
error = mean_squared_error(test['meantemp'], arima_pred)
print('Test MSE: %.3f' % error)
# Plot the results
plt.plot(test.index, test['meantemp'], label='Actual')
plt.plot(test.index, arima_pred, color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Difference in Meantemp')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()


# Fitting the ARIMA model
arima_model = ARIMA(train['meantemp'], order=(2,1,2))
arima_result = arima_model.fit()

# Printing the model summary
print(arima_result.summary())
# Plotting residuals
residuals = arima_result.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.show()

# Checking residual distribution
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=20)
plt.title('Distribution of Residuals')
plt.show()
# Trying different ARIMA configurations
for p in range(3):  # Vary p from 0 to 2
    for d in range(2):  # Vary d from 0 to 1
        for q in range(3):  # Vary q from 0 to 2
            try:
                model = ARIMA(train['meantemp'], order=(p,d,q))
                result = model.fit()
                print(f'ARIMA({p},{d},{q}) - AIC: {result.aic}')
            except:
                continue



# We have seen that simple ARIMA Model is not suitable for this data (There is seasonality in our data)
# Define the SARIMA model parameters
order = (1, 1, 6)  # Non-seasonal order (p, d, q)
seasonal_order = (1, 1, 1, 7)  # Seasonal order (P, D, Q, S)
# Fit the SARIMA model
sarima_model = SARIMAX(endog=train['meantemp'], exog=train[['humidity', 'wind_speed','meanrainfall', 'dewpoint','meanpressure']],
                       order=order, seasonal_order=seasonal_order)
sarima_model_fit = sarima_model.fit()
# Make predictions
sarima_pred = sarima_model_fit.predict(start=test.index[0], end=test.index[-1],
                                            exog=test[['humidity', 'wind_speed','meanrainfall', 'dewpoint','meanpressure']])
# Calculate error
mse = mean_squared_error(test['meantemp'], sarima_pred)
r2 = r2_score(test['meantemp'], sarima_pred)
print('Test MSE:', mse)
print('Test R²: %.3f' % r2)
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(test.index, test['meantemp'], label='Actual')
plt.plot(test.index, sarima_pred, color='red', label='SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Meantemp')
plt.title('SARIMA Forecast')
plt.legend()
plt.show()

# SARIMAX with differenced target (will not be compared with other models)
# df1['meantemp_diff'] = df1['meantemp'].diff().fillna(0)  # diff() default is 1st degree differencing
print(df1)
print(df1.shape)
print(df1.info())
train_size = int(len(df1) * 0.8)
train, test = df1.iloc[:train_size], df1.iloc[train_size:]
print(f'Train shape: {train.shape}')
print(train.head())
print(f'Test shape: {test.shape}')
print(test.head())
print(df1.info())

order = (1, 0, 0)  # Simplify to start with
seasonal_order = (2, 1, 1, 7)  # Weekly seasonality
# Normalize training and testing exogenous variables
scaler = StandardScaler()
train_exog_scaled = scaler.fit_transform(train[['humidity', 'wind_speed', 'meanrainfall', 'dewpoint', 'meanpressure']])
test_exog_scaled = scaler.transform(test[['humidity', 'wind_speed', 'meanrainfall', 'dewpoint', 'meanpressure']])
# Fit the SARIMA model
sarima_model = SARIMAX(endog=train['meantemp_diff'], exog=train_exog_scaled,
                       order=order, seasonal_order=seasonal_order)
sarima_model_fit = sarima_model.fit()
# Make predictions
sarima_pred_diff = sarima_model_fit.predict(start=test.index[0], end=test.index[-1],
                                            exog=test_exog_scaled)
# Cumulative sum to revert differencing, then add last known original value
last_original_value = train['meantemp'].iloc[-1]
sarima_pred = sarima_pred_diff.cumsum() + last_original_value
# Calculate error
mse = mean_squared_error(test['meantemp'], sarima_pred)
r2 = r2_score(test['meantemp'], sarima_pred)
print('Test MSE:', mse)
print('Test R²: %.3f' % r2)
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(test.index, test['meantemp'], label='Actual')
plt.plot(test.index, sarima_pred, color='red', label='SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Meantemp')
plt.title('SARIMA Forecast with differenced target value')
plt.legend()
plt.show()
residuals = sarima_model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('SARIMA Model Residuals')
plt.show()

# # ARIMA-SARIMA with Monthly data
monthly_df = df1.resample('M').mean()
# Split the data into training and testing sets
train_size = int(len(monthly_df) * 0.8)
m_train, m_test = monthly_df.iloc[:train_size], monthly_df.iloc[train_size:]
# ARIMA
# Fit the ARIMA model
arima_model = ARIMA(m_train['meantemp'], order=(1,0,0))  # ARIMA(p,d,q)
arima_model_fit = arima_model.fit()
# Make predictions
arima_pred_m = arima_model_fit.forecast(steps=len(m_test))
# Calculate error
error = mean_squared_error(m_test['meantemp'], arima_pred_m)
print('Test MSE: %.3f' % error)
# Plot the results
plt.plot(m_test.index, m_test['meantemp'], label='Actual')
plt.plot(m_test.index, arima_pred_m, color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Difference in Meantemp')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()

# SARIMA
# Define the SARIMA model parameters
order = (1, 1, 1)  # Non-seasonal order (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, S)
# Fit the SARIMA model
sarima_model = SARIMAX(endog=m_train['meantemp'], exog=m_train[['humidity', 'wind_speed','meanrainfall']],
                       order=order, seasonal_order=seasonal_order)
sarima_model_fit = sarima_model.fit()
# Make predictions
sarima_pred_m = sarima_model_fit.predict(start=m_test.index[0], end=m_test.index[-1],
                                            exog=m_test[['humidity', 'wind_speed','meanrainfall']])
# Calculate error
mse = mean_squared_error(m_test['meantemp'], sarima_pred_m)
r2 = r2_score(m_test['meantemp'], sarima_pred_m)
print('Test MSE:', mse)
print('Test R²: %.3f' % r2)
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(m_test.index, m_test['meantemp'], label='Actual')
plt.plot(m_test.index, sarima_pred_m, color='red', label='SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Meantemp')
plt.title('SARIMA Forecast')
plt.legend()
plt.show()

# from pmdarima import auto_arima
# Fit the model on the training data with optimized settings for monthly seasonality
model = auto_arima(
    m_train['meantemp'],
    seasonal=True,
    m=12,
    max_p=7,
    max_d=1,
    max_q=3,
    max_P=3,
    max_D=1,
    max_Q=2,
    trace=True,    # To print the progress of the fitting
    error_action='ignore',  # Ignore potential errors
    suppress_warnings=True,  # Suppress warnings
    n_jobs=-1  # Use all available CPU cores
)
# Print model summary
print(model.summary())
# Make predictions
n_periods = len(m_test)
auto_arima_pred = model.predict(n_periods=n_periods)
# Calculate R^2 and RMSE
r2 = r2_score(m_test['meantemp'], auto_arima_pred)
rmse = np.sqrt(mean_squared_error(m_test['meantemp'], auto_arima_pred))
print(f'R^2 score: {r2}')
print(f'RMSE: {rmse}')
# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(m_train.index, m_train['meantemp'], label='Train')
plt.plot(m_test.index, m_test['meantemp'], label='Test')
plt.plot(m_test.index, auto_arima_pred, label='Predicted')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Mean Temperature')
plt.title('Mean Temperature Prediction')
plt.show()

# Prophet Model without other features (humidity, wind_speed)
from prophet import Prophet
# Rename columns for Prophet
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'meantemp': 'y'})
# Split the data into training and testing sets
train_size = int(len(df_prophet) * 0.8)
p_train, p_test = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]
# Initialize and fit the Prophet model with additional features
prop_model = Prophet()
prop_model.fit(p_train)
# Make future dataframe and include additional features
future = prop_model.make_future_dataframe(periods=len(p_test), freq='D')
# Make predictions
forecast = prop_model.predict(future)
# Extract the forecasted values for the test period
predicted = forecast[['ds', 'yhat']].set_index('ds').loc[p_test['ds']]
# Calculate error and R²
mse = mean_squared_error(p_test['y'], predicted['yhat'])
r2 = r2_score(p_test['y'], predicted['yhat'])
print('Test MSE: %.3f' % mse)
print('Test R²: %.3f' % r2)
# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
plt.plot(predicted.index, predicted['yhat'], color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Prophet Forecast')
plt.legend()
plt.show()
# Prophet Model using the entire dataset
# Rename columns for Prophet
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'meantemp': 'y'})
# Add additional features
df_prophet['humidity'] = df['humidity'].values
df_prophet['wind_speed'] = df['wind_speed'].values
df_prophet['meanrainfall'] = df['meanrainfall'].values
df_prophet['meanpressure'] = df['meanpressure'].values
df_prophet['dewpoint'] = df['dewpoint'].values
# Split the data into training and testing sets
train_size = int(len(df_prophet) * 0.8)
p_train, p_test = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]
# Initialize and fit the Prophet model with additional features
prop_model = Prophet()
prop_model.add_regressor('humidity')
prop_model.add_regressor('wind_speed')
prop_model.add_regressor('dewpoint')
prop_model.add_regressor('meanpressure')
prop_model.add_regressor('meanrainfall')
prop_model.fit(p_train)
# Make future dataframe and include additional features
future = prop_model.make_future_dataframe(periods=len(p_test), freq='D')
future['humidity'] = df['humidity'].values
future['wind_speed'] = df['wind_speed'].values
future['dewpoint'] = df['dewpoint'].values
future['meanpressure'] = df['meanpressure'].values
future['meanrainfall'] = df['meanrainfall'].values
# Make predictions
forecast = prop_model.predict(future)
# Extract the forecasted values for the test period
predicted = forecast[['ds', 'yhat']].set_index('ds').loc[p_test['ds']]
# Calculate error and R²
mse = mean_squared_error(p_test['y'], predicted['yhat'])
r2 = r2_score(p_test['y'], predicted['yhat'])
print('Test MSE: %.3f' % mse)
print('Test R²: %.3f' % r2)
# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
plt.plot(predicted.index, predicted['yhat'], color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Prophet Forecast')
plt.legend()
plt.show()
# Looking closer to the results
plt.figure(figsize=(10, 5))
plt.plot(p_test['ds'], p_test['y'], label='Actual')
plt.plot(predicted.index, predicted['yhat'], color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Prophet Forecast')
plt.legend()
plt.show()
print(df_prophet.head()) # actual data
print(predicted.head()) # predicted values of p_test
print(forecast.head(3))  # forecast data
# Adding the actual temprature values to the forecast data
forecast['Actual_temp'] = df_prophet['y']
print(forecast.head(3))
plt.figure(figsize=(22,4))
sns.lineplot(data = forecast[['Actual_temp', 'yhat', 'yhat_lower','yhat_upper']])
plt.show()
# there isn't a big gap between yhat_lower and yhat_upper, so I can trust this model
# Compare ARIMA-SARIMA-Prophet Models before Deep Learning Models
print(test.head(2)) # test data from arima sarima models
print(p_test.head(2)) # test data of prophet model
print(arima_pred) # predictions of ARIMA
print(sarima_pred) # predictions of SARIMA
predicted.head() # predictions of prophet
pred_df = test.copy()  # copying test data from arima sarima models (for daily comparision)
pred_df.head()
print(m_test.head()) # monthly test data
m_test["arima_pred_m"] = arima_pred_m.values.round(2)
m_test["sarima_pred_m"] = sarima_pred_m.values.round(2)
# m_test["auto_arima_pred"] = auto_arima_pred.values.round(2)
print(m_test.head())
# Adding predictions of ARIMA, SARIMA, and Prophet
pred_df["arima_pred"] = arima_pred.values.round(2)
pred_df["sarima_pred"] = sarima_pred.values.round(2)
pred_df["prophet_yhat"] = predicted["yhat"].values.round(2)
pred_df.head()
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import math
prophet_mape = mean_absolute_percentage_error(pred_df["meantemp"].values, pred_df["prophet_yhat"].values)
prophet_r = r2_score(pred_df["meantemp"].values, pred_df["prophet_yhat"].values)
prophet_rmse = math.sqrt(mean_squared_error(pred_df["meantemp"].values, pred_df["prophet_yhat"].values))
arima_mape = mean_absolute_percentage_error(pred_df["meantemp"].values, pred_df["arima_pred"].values)
arima_r = r2_score(pred_df["meantemp"].values, pred_df["arima_pred"].values)
arima_rmse = math.sqrt(mean_squared_error(pred_df["meantemp"].values, pred_df["arima_pred"].values))
sarima_mape = mean_absolute_percentage_error(pred_df["meantemp"].values, pred_df["sarima_pred"].values)
sarima_r = r2_score(pred_df["meantemp"].values, pred_df["sarima_pred"].values)
sarima_rmse = math.sqrt(mean_squared_error(pred_df["meantemp"].values, pred_df["sarima_pred"].values))
######## monthly predictions #############
arima_mape_m = mean_absolute_percentage_error(m_test["meantemp"].values, m_test["arima_pred_m"].values)
arima_r_m = r2_score(m_test["meantemp"].values, m_test["arima_pred_m"].values)
arima_rmse_m = math.sqrt(mean_squared_error(m_test["meantemp"].values, m_test["arima_pred_m"].values))
sarima_mape_m = mean_absolute_percentage_error(m_test["meantemp"].values, m_test["sarima_pred_m"].values)
sarima_r_m = r2_score(m_test["meantemp"].values, m_test["sarima_pred_m"].values)
sarima_rmse_m = math.sqrt(mean_squared_error(m_test["meantemp"].values, m_test["sarima_pred_m"].values))

# autoarima_mape_m = mean_absolute_percentage_error(m_test["meantemp"].values, m_test["auto_arima_pred"].values)
# autoarima_r_m = r2_score(m_test["meantemp"].values, m_test["auto_arima_pred"].values)
# autoarima_rmse_m = math.sqrt(mean_squared_error(m_test["meantemp"].values, m_test["auto_arima_pred"].values))

compare_df = {
        'Prophet daily': [prophet_mape,prophet_rmse,prophet_r],
        'ARIMA daily': [arima_mape,arima_rmse,arima_r],
        'SARIMA daily': [sarima_mape,sarima_rmse,sarima_r],
        'ARIMA Monthly': [arima_mape_m,arima_rmse_m,arima_r_m],
        'SARIMA Monthly': [sarima_mape_m,sarima_rmse_m,sarima_r_m],
        # 'autoARIMA Monthly': [autoarima_mape_m,autoarima_rmse_m,autoarima_r_m]
             }
compare_df = pd.DataFrame(compare_df,index=['MAE','RMSE','R2'])
print(compare_df)