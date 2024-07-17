import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.stats import pearsonr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf


import matplotlib
matplotlib.use('TkAgg')  # 指定Agg后端，选择一个合适的后端，如 'TkAgg', 'Qt5Agg', 'Agg', 'MacOSX'
# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data.csv')
data = df.copy()
data = data.set_index('Date')

#zz = data['高频增长因子环比']
zz = data['高频通胀因子环比']
zz_diff = zz.diff(periods = 1)
# 进行ADF检验
result = adfuller(zz)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

result2 = adfuller(zz_diff[1:])
print('ADF Statistic: %f' % result2[0])
print('p-value: %f' % result2[1])
print('Critical Values:')
for key, value in result2[4].items():
    print('\t%s: %.3f' % (key, value))


#model = sm.tsa.arima.ARIMA(zz,order=(1,0,1))
#arima_res=model.fit()
#arima_res.summary()

def get_best_arima_order(series, max_p=10, max_d = 0, max_q=10):
    best_aic = np.inf
    best_order = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for d in range(max_d+1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order
import warnings
# 忽略特定类型的警告
warnings.filterwarnings('ignore')
best_order = get_best_arima_order(zz)
print(best_order)
# 白噪声检验
#acorr_ljungbox(data['高频增长因子环比'], lags = [6, 12],boxpierce=True)
#trend_evaluate = sm.tsa.arma_order_select_ic(zz, ic=['aic', 'bic'], trend='n', max_ar=20,max_ma=1)
#acf = plot_acf(zz)
#plt.title("ACF图")
#plt.show()
# PACF
#pacf=plot_pacf(zz)
#plt.title("PACF图")
#plt.show()

def get_best_arima_order(series, max_p=5, max_d = 0, max_q=5):
    best_aic = np.inf
    best_order = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for d in range(max_d+1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order

def get_arima(data, config):
    window_width = config.window_size
    pre_len = config.pre_len
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

            # 确定最佳ARIMA模型参数
            order = get_best_arima_order(train)
            #order = (1, 0, 1)
            # 生成预测
            predictions = []
            for i in range(pre_len):
                # 使用确定的参数重新初始化模型
                model = ARIMA(train, order=order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)[0]
                predictions.append(forecast)
                train = np.append(train, forecast)  # 将预测值添加到训练集中，进行滚动预测

            # 计算残差并存储
            residuals.extend(test - predictions)
            predictions_avg.extend(predictions)

        # 存储当前列的残差和预测平均值
        residuals_matrix[:, col] = residuals
        predictions_matrix[:, col] = predictions_avg

    return residuals_matrix, predictions_matrix