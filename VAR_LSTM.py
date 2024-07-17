import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import torch
import torch.nn as nn
import torch.optim as optim


def create_lstm_dataset(data, time_steps=1):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_model(X, time_steps, input_dim):
    X = X.reshape((X.shape[0], time_steps, input_dim))
    X = torch.tensor(X, dtype=torch.float32).to(device)
    model = LSTMModel(input_dim=input_dim, hidden_dim=50, output_dim=input_dim, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, X[:, -1, :])
        loss.backward()
        optimizer.step()

    return model


def adf_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': r[0], 'pvalue': r[1], 'n_lags': r[2], 'n_obs': r[3]}
    p_value = output['pvalue']
    if verbose:
        print(f'    Augmented Dickey-Fuller Test on "{name}"')
        print(f'    Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f'    Significance Level = {signif}')
        print(f'    Test Statistic = {output["test_statistic"]}')
        print(f'    No. Lags Chosen = {output["n_lags"]}')

        for key, val in r[4].items():
            print(f'        Critical Value {key} = {val}')

        if p_value <= signif:
            print(f'    => P-Value = {p_value}. Rejecting Null Hypothesis.')
            print(f'    => Series is Stationary.')
        else:
            print(f'    => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.')
            print(f'    => Series is Non-Stationary.')

    return p_value


def difference(data, interval=1):
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def var_lstm_combined_predict(data, var_lags, lstm_time_steps, future_steps):
    # Check for stationarity and differencing if necessary
    differenced_data = data.copy()
    for name, column in data.iteritems():
        p_value = adf_test(column, name=column.name, verbose=True)
        if p_value > 0.05:
            differenced_data[name] = difference(column)

    # Initialize results list
    combined_predictions = []

    # Rolling window prediction
    for i in range(len(differenced_data) - var_lags - future_steps):
        # Rolling window data
        rolling_data = differenced_data[i:i + var_lags]

        # Train VAR model
        var_result = train_var_model(rolling_data, var_lags)
        var_predictions = var_result.fittedvalues
        var_residuals = rolling_data[var_lags:] - var_predictions

        # Prepare LSTM input data
        X = create_lstm_dataset(var_residuals.values, lstm_time_steps)

        # Train LSTM model
        lstm_model = train_lstm_model(X, lstm_time_steps, X.shape[2])

        # Make future predictions
        input_data = var_residuals.values[-lstm_time_steps:]
        future_preds = []

        for _ in range(future_steps):
            input_data_reshaped = torch.tensor(input_data.reshape((1, lstm_time_steps, input_data.shape[1])),
                                               dtype=torch.float32).to(device)
            lstm_model.eval()
            with torch.no_grad():
                lstm_pred = lstm_model(input_data_reshaped).cpu().numpy()
            var_pred = var_result.forecast(data.values[-var_lags:], steps=1)
            combined_pred = var_pred + lstm_pred
            future_preds.append(combined_pred[0])

            input_data = np.vstack([input_data[1:], combined_pred])

        combined_predictions.append(future_preds)

    return np.array(combined_predictions)


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load your multivariate time series data
    # Example: df = pd.read_csv('multivariate_time_series_data.csv')
    df = pd.DataFrame({
        'series1': np.random.randn(150),
        'series2': np.random.randn(150),
        'series3': np.random.randn(150)
    })

    var_lags = 5
    lstm_time_steps = 10
    future_steps = 24

    combined_predictions = var_lstm_combined_predict(df, var_lags, lstm_time_steps, future_steps)
    print(combined_predictions)

