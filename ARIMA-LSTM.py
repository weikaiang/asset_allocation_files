import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from pmdarima import auto_arima
import pmdarima as pm
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import json
from sklearn.preprocessing import MinMaxScaler


def mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return 100 * np.mean(np.abs((actual - prediction)) / actual)

def get_arima(data, window_width, pre_len):
    # 确保输入的数据是一个DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入数据必须是pandas DataFrame")

    # 获取数据的行数和列数
    num_rows, num_cols = data.shape
    # 初始化残差矩阵和预测值平均值矩阵，大小为（总行数 - 滑动窗口大小 - 预测长度 + 1，列数）
    residuals_matrix = np.zeros((num_rows - window_width - pre_len + 1, num_cols))
    predictions_matrix = np.zeros((num_rows - window_width - pre_len + 1, num_cols))

    # 遍历每一列
    for col in range(num_cols):
        # 获取当前列的数据
        series = data.iloc[:, col]
        residuals = []
        predictions_avg = []

        # 滑动窗口方法
        for start in range(num_rows - window_width - pre_len + 1):
            # 训练数据的结束索引
            train_end = start + window_width
            # 测试数据的结束索引
            test_end = train_end + pre_len

            # 获取训练数据
            train = series[start:train_end].values
            # 获取测试数据
            test = series[train_end:test_end].values

            # 初始化模型
            model = auto_arima(train, max_p=3, max_q=3, seasonal=False, trace=False,
                               error_action='ignore', suppress_warnings=True, maxiter=10)
            #这里将ARIMA模型的最大p，q都设置为3，并且没有使用季节模型

            # 确定模型参数
            model.fit(train)
            order = model.get_params()['order']

            # 生成预测
            predictions = []
            for i in range(pre_len):
                # 使用确定的参数重新初始化模型
                model = pm.ARIMA(order=order)
                model.fit(train)
                # 生成单步预测
                predictions.append(model.predict()[0])
                # 将预测值加入训练数据
                train = np.append(train, test[i])

            # 计算残差和预测值平均值
            residuals.append(np.mean(test - np.array(predictions)))
            predictions_avg.append(np.mean(predictions))

        # 将当前列的残差和预测值平均值存入矩阵
        residuals_matrix[:, col] = residuals
        predictions_matrix[:, col] = predictions_avg

    return residuals_matrix, predictions_matrix

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])
        return out


def get_lstm(data, train_len, test_len, lstm_len=4):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    dataset = np.reshape(data.values, (len(data), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    x_train = []
    y_train = []
    x_test = []

    for i in range(lstm_len, train_len):
        x_train.append(dataset_scaled[i - lstm_len:i, 0])
        y_train.append(dataset_scaled[i, 0])
    for i in range(train_len, len(dataset_scaled)):
        x_test.append(dataset_scaled[i - lstm_len:i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()

    # Create the PyTorch model
    model = LSTMModel(input_dim=1, hidden_dim=lstm_len)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    total_loss = 0
    # train
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_train)

        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print training progress
        if (epoch + 1) % 50 == 0:  # 每50轮打印一次
            print(f'Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}')

    # Calculate and print average loss
    average_loss = total_loss / 500
    print(f'Average Loss: {average_loss:.4f}')
    # Prediction
    model.eval()
    predict = model(x_test)
    predict = predict.data.numpy()
    prediction = scaler.inverse_transform(predict).tolist()

    output = []
    for i in range(len(prediction)):
        output.extend(prediction[i])
    prediction = output

    # Error calculation
    mse = mean_squared_error(data.tail(len(prediction)).values, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data.tail(len(prediction)).reset_index(drop=True), pd.Series(prediction))

    return prediction, mse, rmse, mape


def SMA(data, window):
    sma = np.convolve(data[target], np.ones(window), 'same') / window
    return sma


def EMA(data, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[0] = data.iloc[0]  # 设置初始值为序列的第一个值

    for i in range(1, len(data)):
        ema[i] = alpha * data.iloc[i] + (1 - alpha) * ema[i - 1]

    return ema


def WMA(data, window):
    weights = np.arange(1, window + 1)
    wma = np.convolve(data[target], weights / weights.sum(), 'same')
    return wma


# 其他复杂的移动平均技术如 DEMA 可以通过组合上述基础方法实现
# 例如，DEMA 是两个不同窗口大小的 EMA 的组合

if __name__ == '__main__':
    # Load historical data
    # CSV should have columns: ['date', 'OT']
    strpath = '你数据集的地址CSV格式的文件'
    target = 'OT'  # 预测的列明
    data = pd.read_csv('ETTh1.csv', index_col=0, header=0).tail(1500).reset_index(drop=True)[[target]]

    talib_moving_averages = ['SMA']  # 替换你想用的方法

    # 创建一个字典来存储这些函数
    functions = {
        'SMA': SMA,
        # 'EMA': EMA,
        # 'WMA': WMA,
        # 添加其他需要的移动平均函数
    }

    # for ma in talib_moving_averages:
    #     functions[ma] = abstract.Function(ma)

    # Determine kurtosis "K" values for MA period 4-99
    kurtosis_results = {'period': []}
    for i in range(4, 100):
        kurtosis_results['period'].append(i)
        for ma in talib_moving_averages:
            # Run moving average, remove last 252 days (used later for test data set), trim MA result to last 60 days
            ma_output = functions[ma](data[:-252], i)[-60:]
            # Determine kurtosis "K" value
            k = kurtosis(ma_output, fisher=False)

            # add to dictionary
            if ma not in kurtosis_results.keys():
                kurtosis_results[ma] = []
            kurtosis_results[ma].append(k)

    kurtosis_results = pd.DataFrame(kurtosis_results)
    kurtosis_results.to_csv('kurtosis_results.csv')

    # Determine period with K closest to 3 +/-5%
    optimized_period = {}
    for ma in talib_moving_averages:
        difference = np.abs(kurtosis_results[ma] - 3)
        df = pd.DataFrame({'difference': difference, 'period': kurtosis_results['period']})
        df = df.sort_values(by=['difference'], ascending=True).reset_index(drop=True)
        if df.at[0, 'difference'] < 3 * 0.05:
            optimized_period[ma] = df.at[0, 'period']
        else:
            print(ma + ' is not viable, best K greater or less than 3 +/-5%')

    print('\nOptimized periods:', optimized_period)

    simulation = {}
    for ma in optimized_period:
        # Split data into low volatility and high volatility time series
        low_vol = pd.Series(functions[ma](data, optimized_period[ma]))
        high_vol = pd.Series(data[target] - low_vol)

        # Generate ARIMA and LSTM predictions
        print('\nWorking on ' + ma + ' predictions')
        try:
            low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape = get_arima(low_vol, 1000, 252)
        except:
            print('ARIMA error, skipping to next MA type')
            continue

        high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape = get_lstm(high_vol, 1000, 252)

        final_prediction = pd.Series(low_vol_prediction) + pd.Series(high_vol_prediction)
        mse = mean_squared_error(final_prediction.values, data[target].tail(252).values)
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(data[target].tail(252).reset_index(drop=True), final_prediction)

        # Generate prediction accuracy
        actual = data[target].tail(252).values
        df = pd.DataFrame({'real': actual, 'pre': final_prediction}).to_csv('results.csv', index=False)
        result_1 = []
        result_2 = []
        for i in range(1, len(final_prediction)):
            # Compare prediction to previous close price
            if final_prediction[i] > actual[i - 1] and actual[i] > actual[i - 1]:
                result_1.append(1)
            elif final_prediction[i] < actual[i - 1] and actual[i] < actual[i - 1]:
                result_1.append(1)
            else:
                result_1.append(0)

            # Compare prediction to previous prediction
            if final_prediction[i] > final_prediction[i - 1] and actual[i] > actual[i - 1]:
                result_2.append(1)
            elif final_prediction[i] < final_prediction[i - 1] and actual[i] < actual[i - 1]:
                result_2.append(1)
            else:
                result_2.append(0)

        accuracy_1 = np.mean(result_1)
        accuracy_2 = np.mean(result_2)

        simulation[ma] = {'low_vol': {'prediction': low_vol_prediction, 'mse': low_vol_mse,
                                      'rmse': low_vol_rmse, 'mape': low_vol_mape},
                          'high_vol': {'prediction': high_vol_prediction, 'mse': high_vol_mse,
                                       'rmse': high_vol_rmse},
                          'final': {'prediction': final_prediction.values.tolist(), 'mse': mse,
                                    'rmse': rmse, 'mape': mape},
                          'accuracy': {'prediction vs close': accuracy_1, 'prediction vs prediction': accuracy_2}}

        # save simulation data here as checkpoint
        with open('simulation_data.json', 'w') as fp:
            json.dump(simulation, fp)

    for ma in simulation.keys():
        print('\n' + ma)
        print('Prediction vs Close:\t\t' + str(round(100 * simulation[ma]['accuracy']['prediction vs close'], 2))
              + '% Accuracy')
        print(
            'Prediction vs Prediction:\t' + str(round(100 * simulation[ma]['accuracy']['prediction vs prediction'], 2))
            + '% Accuracy')
        print('MSE:\t', simulation[ma]['final']['mse'],
              '\nRMSE:\t', simulation[ma]['final']['rmse'],
              '\nMAPE:\t', simulation[ma]['final']['mape'])