import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def get_var(data, window_width=60, pre_len=15):
    # 确保输入的数据是一个DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入数据必须是pandas DataFrame")

    # 获取数据的行数和列数
    num_rows, num_cols = data.shape
    # 初始化残差矩阵，大小为（总行数 - 滑动窗口大小 - 预测长度 + 1，列数）
    residuals_matrix = np.zeros((num_rows - window_width - pre_len + 1, num_cols))

    # 滑动窗口方法
    for start in range(num_rows - window_width - pre_len + 1):
        # 训练数据的结束索引
        train_end = start + window_width
        # 测试数据的结束索引
        test_end = train_end + pre_len

        # 获取训练数据
        train = data.iloc[start:train_end]
        # 获取测试数据
        test = data.iloc[train_end:test_end]

        # 动态设置maxlags，确保其值小于观察数的四分之一
        maxlags = min(15, train.shape[0] // 4)

        # 初始化VAR模型并拟合
        model = VAR(train)
        fitted_model = model.fit(maxlags, ic='aic')

        # 生成预测
        predictions = fitted_model.forecast(train.values[-fitted_model.k_ar:], steps=pre_len)

        # 计算残差
        residuals = test.values - predictions
        residuals_mean = np.mean(residuals, axis=0)

        # 将当前窗口的残差存入残差矩阵
        residuals_matrix[start] = residuals_mean

    return residuals_matrix

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    cols_data = df.columns[1:]  # 除去时间列之后的index值
    df_data = df[cols_data]  # 获取除时间列之外的数据
    residuals_matrix = get_var(df_data)