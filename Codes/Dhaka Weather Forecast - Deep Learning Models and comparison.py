import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import gc

df = pd.read_csv("Data/dhakaweathertrain.csv",
                 parse_dates=['date'],  # change to date time format
                 index_col="date")
df1 = df.copy()
df = df.rename(columns={'temp': 'meantemp'})
sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
gc.collect()
print(df.head())
print(df.info())
print(df.describe())
# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
dl_train, dl_test = df.iloc[:train_size], df.iloc[train_size:]
print(f"Training data size: {len(dl_train)}, Test data size: {len(dl_test)}")
# find out what scaling to perform
f_columns = ['meantemp', 'humidity', 'wind_speed', 'meanrainfall', 'dewpoint', 'meanpressure']
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
sns.boxplot(y=df['meantemp'])
plt.title('Boxplot of meantemp')
plt.subplot(2, 3, 2)
sns.boxplot(y=df['humidity'])
plt.title('Boxplot of humidity')
plt.subplot(2, 3, 3)
sns.boxplot(y=df['wind_speed'])
plt.title('Boxplot of wind_speed')
plt.subplot(2, 3, 4)
sns.boxplot(y=df['meanrainfall'])
plt.title('Boxplot of meanrainfall')
plt.subplot(2, 3, 5)
sns.boxplot(y=df['dewpoint'])
plt.title('Boxplot of dewpoint')
plt.subplot(2, 3, 6)
sns.boxplot(y=df['meanpressure'])
plt.title('Boxplot of meanpressure')
plt.tight_layout()
plt.show()
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
from sklearn.preprocessing import RobustScaler, MinMaxScaler
robust_scaler = RobustScaler()   # scaler for wind_speed
minmax_scaler = MinMaxScaler()  # scaler for humidity
target_transformer = MinMaxScaler()   # scaler for target (meantemp)
dl_train['wind_speed'] = robust_scaler.fit_transform(dl_train[['wind_speed']])  # robust for wind_speed
dl_train['humidity'] = robust_scaler.fit_transform(dl_train[['humidity']])
dl_train['meanrainfall'] = robust_scaler.fit_transform(dl_train[['meanrainfall']])
dl_train['dewpoint'] = minmax_scaler.fit_transform(dl_train[['dewpoint']]) # minmax for humidity
dl_train['meanpressure'] = minmax_scaler.fit_transform(dl_train[['meanpressure']]) # minmax for humidity
dl_train['meantemp'] = target_transformer.fit_transform(dl_train[['meantemp']]) # target
dl_test['wind_speed'] = robust_scaler.fit_transform(dl_test[['wind_speed']])  # robust for wind_speed
dl_test['humidity'] = robust_scaler.fit_transform(dl_test[['humidity']])
dl_test['meanrainfall'] = robust_scaler.fit_transform(dl_test[['meanrainfall']])
dl_test['dewpoint'] = minmax_scaler.fit_transform(dl_test[['dewpoint']]) # minmax for humidity
dl_test['meanpressure'] = minmax_scaler.fit_transform(dl_test[['meanpressure']]) # minmax for humidity
dl_test['meantemp'] = target_transformer.fit_transform(dl_test[['meantemp']]) # target
print(df.head())
print(dl_train.head())
print(dl_test.head())


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.python.keras.callbacks import EarlyStopping

# Function to create sequences for RNN, LSTM, GRU
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Feature and target selection
predictors = ['wind_speed', 'humidity', 'dewpoint', 'meanrainfall', 'meanpressure']
target = 'meantemp'
#Create sequences
sequence_length = 3
X_train, y_train = create_dataset(dl_train[predictors], dl_train[target], sequence_length)
X_test, y_test = create_dataset(dl_test[predictors], dl_test[target], sequence_length)
#
# Build the SimpleRNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
rnn_model.add(Dense(1))  # Output layer for regression
rnn_model.compile(optimizer='adam', loss='mse')

# Model summary
print(rnn_model.summary())

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = rnn_model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=[early_stopping]
)

# Evaluate the model
loss = rnn_model.evaluate(X_test, y_test)
print(f'Validation Loss: {loss}')

# Make predictions
predictions = rnn_model.predict(X_test)

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Mean Temp', color='blue')
plt.plot(predictions, label='Predicted Mean Temp', color='red')
plt.title('Mean Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

# Get training and validation losses from history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot loss values over epochs
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#LSTM
# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
lstm_model.add(Dense(1))  # Output layer for regression
lstm_model.compile(optimizer='adam', loss='mse')

# Model summary
print(lstm_model.summary())

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=[early_stopping]
)

# Evaluate the model
loss = lstm_model.evaluate(X_test, y_test)
print(f'Validation Loss: {loss}')

# Make predictions
predictions = lstm_model.predict(X_test)

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Mean Temp', color='blue')
plt.plot(predictions, label='Predicted Mean Temp', color='red')
plt.title('Mean Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

# Get training and validation losses from history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot loss values over epochs
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#GRU
# Build the GRU model
gru_model = Sequential()
gru_model.add(GRU(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
gru_model.add(Dense(1))  # Output layer for regression
gru_model.compile(optimizer='adam', loss='mse')

# Model summary
print(gru_model.summary())

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = gru_model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=[early_stopping]
)

# Evaluate the model
loss = gru_model.evaluate(X_test, y_test)
print(f'Validation Loss: {loss}')

# Make predictions
predictions = gru_model.predict(X_test)

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Mean Temp', color='blue')
plt.plot(predictions, label='Predicted Mean Temp', color='red')
plt.title('Mean Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

# Get training and validation losses from history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot loss values over epochs
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
#

# Create sequences
sequence_length = 3
X_train, y_train = create_dataset(dl_train[predictors], dl_train[target], sequence_length)
X_test, y_test = create_dataset(dl_test[predictors], dl_test[target], sequence_length)

# Function to build and train a model
def build_and_train_model(model_type='SimpleRNN'):
    # Build the model based on the type
    model = Sequential()
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
    elif model_type == 'LSTM':
        model.add(LSTM(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
    elif model_type == 'GRU':
        model.add(GRU(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse')

    # Model summary
    print(f"\n{model_type} Model Summary:")
    print(model.summary())

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        batch_size=16,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'{model_type} Validation Loss: {loss}')

    # Make predictions
    predictions = model.predict(X_test)

    # Plot predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Mean Temp', color='blue')
    plt.plot(predictions, label='Predicted Mean Temp', color='red')
    plt.title(f'{model_type} Mean Temperature Prediction')
    plt.xlabel('Time')
    plt.ylabel('Mean Temperature')
    plt.legend()
    plt.show()

    # Get training and validation losses from history
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Plot loss values over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_type} Training and Validation Loss')
    plt.legend()
    plt.show()

# Run all models
for model_type in ['SimpleRNN', 'LSTM', 'GRU']:
    build_and_train_model(model_type)

# Function to create sequences for RNN, LSTM, GRU
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Feature and target selection
predictors = ['wind_speed', 'humidity', 'dewpoint', 'meanrainfall', 'meanpressure']
target = 'meantemp'
# Create sequences
sequence_length = 3
X_train, y_train = create_dataset(dl_train[predictors], dl_train[target], sequence_length)
X_test, y_test = create_dataset(dl_test[predictors], dl_test[target], sequence_length)
#

# Function to build and train a model
def build_and_train_model(model_type='SimpleRNN'):
    # Build the model based on the type
    model = Sequential()
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
    elif model_type == 'LSTM':
        model.add(LSTM(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
    elif model_type == 'GRU':
        model.add(GRU(100, activation='tanh', input_shape=(sequence_length, len(predictors))))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse')

    # Model summary
    print(f"\n{model_type} Model Summary:")
    print(model.summary())

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        batch_size=16,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'{model_type} Validation Loss: {loss}')

    # Make predictions
    predictions = model.predict(X_test)

    # Compute the performance metrics
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    return predictions, mape, r2, rmse


# Initialize dictionary to store results
dl_compare = {}

# Run all models
for model_type in ['SimpleRNN', 'LSTM', 'GRU']:
    # Get predictions and metrics for each model
    predictions, mape, r2, rmse = build_and_train_model(model_type)

    # Store the metrics in the dictionary
    dl_compare[f'{model_type} daily'] = [mape, rmse, r2]

# Convert the dictionary to a DataFrame
dl_compare = pd.DataFrame(dl_compare, index=['MAPE', 'RMSE', 'R2'])

# Concatenate dl_compare with compare_df
compare_models_df = pd.concat([compare_df, dl_compare], axis=1) # compare_df comes from Dhaka Weather Forecast - ARIMA_SARIMA_Prophet.py

# Print the final comparison DataFrame
print(compare_models_df)

# Optionally, you can plot the performance metrics
compare_models_df.T.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison: MAPE, RMSE, R2')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comparison DataFrame with your model metrics
data = {
    'Model': ['Prophet daily', 'ARIMA daily', 'SARIMA daily', 'RNN daily', 'LSTM daily', 'BiLSTM daily', 'GRU daily'],
    'MAE': [0.052, 0.245, 0.143, 0.045, 0.046, 0.045, 0.046],  # Replace with actual MAE values
    'RMSE': [1.718, 8.100, 4.803, 1.642, 1.644, 1.637, 1.648],  # Replace with actual RMSE values
    'R2': [0.908, -1.050, 0.279, 0.916, 0.916, 0.917, 0.915]    # Replace with actual R2 values
}

# Create a DataFrame from the data
compare = pd.DataFrame(data)

# Function to label the bars with values
def labels(ax):
    for p in ax.patches:
        width = p.get_width()  # The value of the bar (for vertical bars)
        ax.text(width,           # Position text at the end of each bar
                p.get_y() + p.get_height() / 2,  # Position text at the center of the bar
                '{:.3f}'.format(width),  # Format text to 3 decimal places
                ha='left',               # Align text to the left
                va='center')             # Align text to the center vertically

# Set the figure size
plt.figure(figsize=(15, 14))

# Plot for R2 Scores
plt.subplot(411)
compare = compare.sort_values(by="R2", ascending=False)
ax = sns.barplot(x="R2", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title('R2 Scores')

# Plot for MAE Scores
plt.subplot(412)
compare = compare.sort_values(by="MAE", ascending=False)
ax = sns.barplot(x="MAE", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title('MAE Scores')

# Plot for RMSE Scores
plt.subplot(413)
compare = compare.sort_values(by="RMSE", ascending=False)
ax = sns.barplot(x="RMSE", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title('RMSE Scores')

# Adjust the layout to make sure everything fits
plt.tight_layout()

# Show the plot
plt.show()

