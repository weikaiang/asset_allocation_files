import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
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
    df = config.data_df
    pre_len = config.pre_len
    train_window = config.window_size

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    true_data = df_data.values

    true_data_list = []
    for i in range(config.input_size):
        init = 10000
        list_temp = []
        for j in range(len(true_data)):
            factor = (1 + true_data[j, i]) * init
            init = factor
            list_temp.append(init)
        true_data_list.append(list_temp)
    true_data_fac = np.array(true_data_list).transpose()
    factor_df = pd.DataFrame(true_data_fac)

    scaler_fac = StandardScaler()
    fac_data_normalized = scaler_fac.fit_transform(true_data_fac)
    fac_data_normalized = torch.FloatTensor(fac_data_normalized).to(device)
    fac_inout_seq = create_inout_sequences(fac_data_normalized, train_window, pre_len, config)
    fac_dataset = TimeSeriesDataset(fac_inout_seq)
    fac_loader = DataLoader(fac_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    scaler_train = StandardScaler()
    scaler_valid = StandardScaler()
    scaler_test = StandardScaler()

    train_data = true_data
    valid_data = true_data[int(0.15 * len(true_data)):int(0.30 * len(true_data))]
    test_data = true_data[:int(0.15 * len(true_data))]
    print("训练集尺寸:", len(train_data))

    train_data_normalized = scaler_train.fit_transform(train_data)
    test_data_normalized = scaler_test.fit_transform(test_data)
    valid_data_normalized = scaler_valid.fit_transform(valid_data)

    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)

    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, scaler_train, df_data, factor_df, fac_loader


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

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

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
                            batch_first=True)
        self.hidden_size = hidden_size
        self.obs_len = obs_len
        self.output_horizon = output_horizon
        self.attention = SelfAttention(input_size, output_horizon)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers
        self.device = device

    def forward(self, x):
        x = self.attention(x, x, x, None)

        batch_size, obs_len, features_size = x.shape

        xconcat = self.hidden(x)

        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(self.device)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht[-1, :, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)
        ypred = self.linear(H)
        ypred = ypred[:, -self.obs_len:, :]
        return ypred


def predict(model, args, device, scaler, data):
    df = data.copy()
    df = df.iloc[:, 0:][-args.window_size:].values
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)
    model.load_state_dict(torch.load('save_model.pth', map_location=device))
    model.eval()

    pred = model(tensor_pred)[0]

    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    pred_df = pd.DataFrame(pred)

    return pred_df


class DTWLoss(nn.Module):
    def __init__(self):
        super(DTWLoss, self).__init__()

    def forward(self, s1_batch, s2_batch):
        device = s1_batch.device

        batch_size = s1_batch.shape[0]
        len1 = s1_batch.shape[1]
        len2 = s2_batch.shape[1]
        feat_dim = s1_batch.shape[2]

        DTW = torch.zeros((batch_size, len1, len2), device=device) + float('inf')

        DTW[:, 0, 0] = torch.sum((s1_batch[:, 0] - s2_batch[:, 0]) ** 2, dim=-1)
        for i in range(1, len1):
            DTW[:, i, 0] = DTW[:, i - 1, 0] + torch.sum((s1_batch[:, i] - s2_batch[:, 0]) ** 2, dim=-1)
        for j in range(1, len2):
            DTW[:, 0, j] = DTW[:, 0, j - 1] + torch.sum((s1_batch[:, 0] - s2_batch[:, j]) ** 2, dim=-1)
        for i in range(1, len1):
            for j in range(1, len2):
                cost = torch.sum((s1_batch[:, i] - s2_batch[:, j]) ** 2, dim=-1)
                DTW[:, i, j] = cost + torch.min(torch.stack([DTW[:, i - 1, j],
                                                             DTW[:, i, j - 1],
                                                             DTW[:, i - 1, j - 1]], dim=-1), dim=-1)[0]

        loss = DTW[:, -1, -1]
        return torch.mean(loss)


def train_model(args, device, model):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>开始训练<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    start = time.time()
    train_loader, scaler, data, factor_df, fac_loader = create_dataloader(args, device)
    loss_fn = nn.MSELoss()
    model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in tqdm(range(args.epochs)):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'save_model.pth')
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>训练结束<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    end = time.time()
    print("训练时间:", end - start)

def LSTMargs(data_df, method='rate', window_size=60, pre_len=15, size=8, lr=0.001, drop_out=0.05, epochs=50,
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
    parser.add_argument('-method', type=str, default=method, help="loss计算方式")  # 可能需要调整

    # model
    parser.add_argument('-hidden-size', type=int, default=128, help="隐藏层单元数")
    parser.add_argument('-kernel-sizes', type=str, default='3')
    parser.add_argument('-laryer_num', type=int, default=1)
    # device
    #parser.add_argument('-use_gpu', type=bool, default=False)
    #parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args_LSTM = parser.parse_args()

    return args_LSTM

def pred(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, scaler_train, rate_df, fac_df, fac_loader = create_dataloader(args, device)
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
        train_model(args, device, model)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        pred = predict(model, args, device, scaler_train, rate_df)
        #pred_rate = getFactorRate(fac_df, pred, args)
    return pred

if __name__ == '__main__':
    data = pd.read_csv('asset.csv')
    # args1 = LSTMargs(data, epochs = 30)
    # pred_val = pred(args1)
    # pred_val.to_csv('pred_rate.csv')
    args2 = LSTMargs(data, epochs=50)
    pred = pred(args2)
