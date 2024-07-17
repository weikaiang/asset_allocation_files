import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

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


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)  # 填你自己的数据地址,自动选取你最后一列数据为特征列 # 添加你想要预测的特征列
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    # 将特征列移到末尾
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values

    # 定义标准化优化器
    scaler_train = StandardScaler()
    scaler_valid = StandardScaler()
    scaler_test = StandardScaler()

    train_data = true_data[int(0.3 * len(true_data)):]
    valid_data = true_data[int(0.15 * len(true_data)):int(0.30 * len(true_data))]
    test_data = true_data[:int(0.15 * len(true_data))]
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))

    # 进行标准化处理
    train_data_normalized = scaler_train.fit_transform(train_data)
    test_data_normalized = scaler_test.fit_transform(test_data)
    valid_data_normalized = scaler_valid.fit_transform(valid_data)

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader, scaler_train, scaler_test, scaler_valid


class LSTM_GRU(nn.Module):
    def __init__(self, args, device):
        super(LSTM_GRU, self).__init__()
        self.args = args
        self.device = device
        self.dropout = nn.Dropout(args.drop_out)

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, batch_first=True)
        self.gru = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=1, batch_first=True)
        self.linearOut = nn.Linear(args.hidden_size, args.input_size)

    def forward(self, x):
        hidden = ((torch.zeros(1, x.size(0), self.args.hidden_size).to(self.device)),
                  (torch.zeros(1, x.size(0), self.args.hidden_size).to(self.device)))
        x, lstm_h = self.lstm(x, hidden)
        x = self.dropout(x)
        x = torch.tanh(torch.transpose(x, 1, 2))
        x = x.permute(0, 2, 1)
        x, gru_ = self.gru(x)
        x = self.dropout(x)
        x = torch.tanh(torch.transpose(x, 1, 2))
        x = x.permute(0, 2, 1)
        x = self.linearOut(x)
        x = x[:, -args.pre_len:, :]

        return x


def train(model, args, device):
    start_time = time.time()  # 计算起始时间
    lstm_model = model
    loss_function = nn.MSELoss()
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
        torch.save(lstm_model.state_dict(), 'save_model.pth')
        time.sleep(0.1)

    # 保存模型

    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    return scaler_train

def predict(model, args, device, scaler):
    # 预测未知数据的功能
    # 重新读取数据
    df = pd.read_csv(args.data_path)
    train_data = df[[args.target]][int(0.3 * len(df)):]
    df = df.iloc[:, 1:][-args.window_size:].values  # 转换为nadarry
    scaler_tr = StandardScaler()
    scaler_tr.fit_transform(train_data.values)
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式

    pred = model(tensor_pred)[0]

    if args.feature == 'M' or args.feature == 'S':
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
    else:
        pred = scaler_tr.inverse_transform(pred.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='LSTM-Attention', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=128, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=24, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='ETTh1Test.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='OT', help='你需要预测的特征列，这个值会最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=7, help='你的特征个数不算时间那一列')
    parser.add_argument('-output_size', type=int, default=7, help='输出特征个数只有两种选择和你的输入特征一样即输入多少输出多少，另一种就是多元预测单元')
    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=20, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=16, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')

    # model
    parser.add_argument('-hidden-size', type=int, default=128, help="隐藏层单元数")
    parser.add_argument('-kernel-sizes', type=str, default='3')
    parser.add_argument('-laryer_num', type=int, default=1)
    # device
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args = parser.parse_args()

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print(device)
    train_loader, test_loader, valid_loader, scaler_train, scaler_test, scaler_valid = create_dataloader(args, device)

    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = LSTM_GRU(args, device).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, device)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler_train)