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

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS' or config.feature == 'S':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # df = pd.read_csv(config.data_path)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    df = config.data_df
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    cols_data = df.columns[1:]  # 除去时间列之后的index值
    df_data = df[cols_data]  # 获取除时间列之外的数据
    # 留下最后一个预测长度为测试集，其余为训练集
    df_data_test = df_data[-config.pre_len:]
    df_data = df_data[:-config.pre_len]

    df_data, arima_pred = get_arima(df_data, config)

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values

    # 定义标准化优化器
    scaler_train = StandardScaler()

    # 保留下的数据，全部作为测试集
    train_data = true_data
    print("训练集尺寸:", len(train_data))

    # 进行标准化处理
    train_data_normalized = scaler_train.fit_transform(train_data)
    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, scaler_train, df_data, df_data_test, arima_pred

import warnings
# 忽略特定类型的警告
warnings.filterwarnings('ignore')


def get_best_arima_order(series, max_p=5, max_d=0, max_q=5):
    best_aic = np.inf
    best_order = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for d in range(max_d + 1):
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
    pre_len = config.pre_len
    # 确保输入的数据是一个DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入数据必须是pandas DataFrame")

    # 获取数据的行数和列数
    num_rows, num_cols = data.shape
    # 初始化残差矩阵和预测值矩阵
    residuals_matrix = np.zeros((num_rows - pre_len, num_cols))
    predictions_matrix = np.zeros((pre_len, num_cols))

    # 遍历每一列
    for col in range(num_cols):
        # 获取当前列的数据
        series = data.iloc[:, col]

        # 确定最佳ARIMA模型参数
        order = get_best_arima_order(series)

        # 训练ARIMA模型
        model = ARIMA(series, order=order)
        model_fit = model.fit()

        # 预测未来 pre_len 步
        forecast = model_fit.forecast(steps=pre_len)

        # 计算残差
        residuals = series[:-pre_len] - model_fit.fittedvalues[:-(pre_len)]

        # 存储残差和预测值
        residuals_matrix[:, col] = residuals
        predictions_matrix[:, col] = forecast
    #将array数据类型转化为dataframe
    residuals_matrix, predictions_matrix = pd.DataFrame(residuals_matrix), pd.DataFrame(predictions_matrix)
    return residuals_matrix, predictions_matrix

class SelfAttention(nn.Module):
    def __init__(self, feature_size, heads):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.head_dim = feature_size // heads

        assert (
                self.head_dim * heads == feature_size
        ), "Feature size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, feature_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do batch matrix multiplication
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.feature_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TPALSTM(nn.Module):
    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers, device):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bias=True,
                            batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.obs_len = obs_len
        self.output_horizon = output_horizon
        self.attention = SelfAttention(input_size, output_horizon)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers
        self.device = device

    def forward(self, x):

        x = self.attention(x, x, x, None)

        batch_size, obs_len, features_size = x.shape  # (batch_size, obs_len, features_size)

        xconcat = self.hidden(x)  # (batch_size, obs_len, hidden_size)

        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(
            self.device)  # (batch_size, obs_len-1, hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            self.device)  # (num_layers, batch_size, hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
            out, (ht, ct) = self.lstm(xt, (ht, ct))  # ht size (num_layers, batch_size, hidden_size)
            htt = ht[-1, :, :]  # (batch_size, hidden_size)
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)  # (batch_size, obs_len-1, hidden_size)
        ypred = self.linear(H)  # (batch_size, output_horizon)
        ypred = ypred[:, -self.obs_len:, :]
        return ypred

def predict(model, args, device, scaler, data):
    # 预测未知数据的功能
    # 重新读取数据
    # df = pd.read_csv(args.data_path)
    # df = args.data_df
    df = data.copy()
    # train_data = df[[args.target]][int(0.3 * len(df)):]
    df = df.iloc[:, 0:][-args.window_size:].values  # 转换为nadarry
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    # method = args.method
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式

    pred = model(tensor_pred)[0]

    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    # 储存预测数据进入dataframe
    pred_df = pd.DataFrame(pred)

    return pred_df

class DTWLoss(nn.Module):
    def __init__(self):
        super(DTWLoss, self).__init__()

    def forward(self, s1_batch, s2_batch):
        device = s1_batch.device  # Get the device (CPU or GPU) of the input tensor

        batch_size = s1_batch.shape[0]
        len1 = s1_batch.shape[1]
        len2 = s2_batch.shape[1]
        feat_dim = s1_batch.shape[2]  # Assuming both s1_batch and s2_batch have the same feature dimension

        # Initialize DTW matrix with infinities for each batch element
        DTW = torch.zeros((batch_size, len1, len2), device= device) + float('inf')

        # Set the initial conditions for each batch element
        DTW[:, 0, 0] = torch.sum((s1_batch[:, 0] - s2_batch[:, 0]).pow(2), dim=1)

        # Fill the DTW matrix
        for i in range(len1):
            for j in range(len2):
                if i > 0 or j > 0:
                    cost = torch.sum((s1_batch[:, i] - s2_batch[:, j]).pow(2), dim=1)
                    prev_costs = []

                    if i > 0:
                        prev_costs.append(DTW[:, i - 1, j])
                    if j > 0:
                        prev_costs.append(DTW[:, i, j - 1])
                    if i > 0 and j > 0:
                        prev_costs.append(DTW[:, i - 1, j - 1])

                    prev_costs_tensor = torch.stack(prev_costs, dim=1)
                    min_prev, _ = torch.min(prev_costs_tensor, dim=1)

                    DTW[:, i, j] = cost + min_prev

        # Return the square root of the minimum DTW distance for each batch element
        dtw_loss = torch.mean(torch.sqrt(DTW[:, len1 - 1, len2 - 1]))
        return dtw_loss

class ATPLRLoss(nn.Module):
    def __init__(self):
        super(ATPLRLoss, self).__init__()

    def forward(self, s1_batch, s2_batch):
        def adaptive_window_plr(s):
            batch_size = s.shape[0]
            seq_length = s.shape[1]
            feat_dim = s.shape[2]

            max_window_size = seq_length // 2  # Max window size is half of sequence length

            # Initialize list to store segment means
            segments = []

            # Calculate cumulative sums for mean calculation
            cumulative_sum = torch.cumsum(s, dim=1)

            # Initialize window size and start index
            window_size = 1
            start = 0

            while start < seq_length:
                # Calculate end index of current window
                end = min(start + window_size, seq_length)

                # Calculate mean for current window
                segment_mean = (cumulative_sum[:, end - 1] - cumulative_sum[:, start - 1]) / (end - start + 1)
                segments.append(segment_mean.view(batch_size, 1, feat_dim))

                # Update start index
                start += window_size

                # Increase window size exponentially
                window_size *= 2
                if window_size > max_window_size:
                    window_size = max_window_size

            # Stack all segments
            return torch.cat(segments, dim=1)
        # Get adaptive window piecewise linear representations
        aw_plr_s1 = adaptive_window_plr(s1_batch)
        aw_plr_s2 = adaptive_window_plr(s2_batch)

        # Calculate squared differences
        diff = (aw_plr_s1 - aw_plr_s2).pow(2)

        # Sum of squared differences
        aw_plr_distance = torch.mean(diff, dim=1)  # Mean over segments

        # Return the mean AW-PLR distance as the loss (the lower the similarity, the higher the loss)
        AWPLRLoss = torch.mean(aw_plr_distance)
        return AWPLRLoss

def train(model, args, train_loader):
    start_time = time.time()  # 计算起始时间,用于展示进度
    lstm_model = model
    #选择损失函数
    loss = args.loss
    loss_select ={
        'MSE': nn.MSELoss(),
        'DTW': DTWLoss(),
        'PLR': ATPLRLoss(),
    }
    loss_function = loss_select[loss]
    #loss_function = nn.MSELoss()
    #loss_function = DTWLoss()
    #loss_function = ATPLRLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    epochs = args.epochs
    lstm_model.train()  # 训练模式
    results_loss = []
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            lstm_model.train()

            optimizer.zero_grad()

            y_pred = lstm_model(seq)

            single_loss = loss_function(y_pred, labels)

            single_loss.backward()

            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))
        torch.save(lstm_model.state_dict(), 'save_model.pth')  # 将模型储存在目录下
        time.sleep(0.1)

    # 保存模型
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")


def calculate_ic_ir(time_series1, time_series2):

    # 计算IC（Information Coefficient）
    ic, _ = pearsonr(time_series1, time_series2)

    # 计算时间序列的平均值和标准差
    mean1, mean2 = np.mean(time_series1), np.mean(time_series2)
    std1, std2 = np.std(time_series1), np.std(time_series2)

    # 计算IR（Information Ratio）
    ir = (mean1 - mean2) / np.sqrt(std1 ** 2 + std2 ** 2)

    return ic, ir

def evaluate(model, args, device, scaler, data, test_data):
    # 预测未知数据的功能
    # 重新读取数据
    # df = pd.read_csv(args.data_path)
    # df = args.data_df
    df = data.copy()
    # train_data = df[[args.target]][int(0.3 * len(df)):]
    df_pred = df.iloc[:, 0:][-args.window_size:].values  # 取训练数据倒数第一个窗口，转换为nadarry
    real = test_data.values  # 取所有数据最后一个预测长度的真实数据，数据格式为nadarry
    pre_data = scaler.transform(df_pred)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    # method = args.method
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式

    pred = model(tensor_pred)[0]

    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    IC = []
    IR = []
    for i in range(pred.shape[1]):
        # 计算每个因子的IC和IR值，并且返回
        ic_temp, ir_temp = calculate_ic_ir(pred[:, i], real[:, i])
        IC.append(ic_temp)
        IR.append(ir_temp)

    return IC, IR

def LSTMargs(data_df, method='rate', loss = 'MSE', window_size=60, pre_len=15, size=8, lr=0.001, drop_out=0.05, epochs=50,
             batch_size=16):
    """
    window_size:用来预测的时间窗口大小
    pre_len:预测未来数据长度
    data_path:读取历史数据位置, csv
    data_df:读取的历史数据, pd.DataFrame
    size:因子/资产个数
    lr:学习率
    drop_out:随机丢弃概率
    epochs:训练轮次
    batch_size:批次大小
    loss:损失函数选择
    """

    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='LSTM-Attention', help="")
    parser.add_argument('-window_size', type=int, default=window_size, help="时间窗口大小, window_size > pre_len")  # 可能需要调整
    parser.add_argument('-pre_len', type=int, default=pre_len, help="预测未来数据长度")  # 可能需要调整
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help="是否打乱数据加载器中的数据顺序")
    # parser.add_argument('-data_path', type=str, default = data_path, help="你的数据地址") #可能需要调整
    parser.add_argument('-data_df', type=str, default=data_df, help="你的数据")  # 可能需要调整
    parser.add_argument('-input_size', type=int, default=size, help='你的特征个数不算时间那一列')  # 可能需要调整
    parser.add_argument('-output_size', type=int, default=size,
                        help='输出特征个数只有两种选择和你的输入特征一样即输入多少输出多少，另一种就是多元预测单元')  # 可能需要调整

    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')

    # learning 模型具体参数调整
    parser.add_argument('-lr', type=float, default=lr, help="学习率")
    parser.add_argument('-drop_out', type=float, default=drop_out, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=epochs, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=batch_size, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')
    parser.add_argument('-method', type=str, default=method, help="数据格式")  # 可能需要调整

    # model
    parser.add_argument('-hidden-size', type=int, default=128, help="隐藏层单元数")
    parser.add_argument('-kernel-sizes', type=str, default='3')
    parser.add_argument('-laryer_num', type=int, default=1)
    parser.add_argument('-loss', type=str, default=loss, help = "损失函数类型")
    # device
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args_LSTM = parser.parse_args()

    return args_LSTM


def pred(args):
    device = torch.device("cpu")
    #train_loader
    train_loader, scaler_train, rate_df, test_data, arima_pred = create_dataloader(args, device)
    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = TPALSTM(args.input_size, args.output_size, args.hidden_size, args.pre_len, args.laryer_num, device).to(
            device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, train_loader)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        lstm_pred = predict(model, args, device, scaler_train, rate_df)
        prediction = arima_pred.add(lstm_pred) #将残差和线性预测结果相加，得到最后结果
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>输出最后一个窗口的IC和IR值<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        ic, ir = evaluate(model, args, device, scaler_train, rate_df, test_data)
        print(f"IC: {ic}, IR: {ir}")
        print(prediction)
    return prediction, ic, ir


# 主要参数修改#
if __name__ == '__main__':

    data = pd.read_csv('assetfac.csv')
    # args1 = LSTMargs(data, epochs = 30)
    # pred_val = pred(args1)
    # pred_val.to_csv('pred_rate.csv')
    args2 = LSTMargs(data, epochs=50, window_size=504, pre_len=60, loss = 'PLR')
    pred, ic, ir= pred(args2)
    # pred_fac.to_csv('pred_fac.csv')
