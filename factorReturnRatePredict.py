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
# # 随机数种子
# np.random.seed(1)

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
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq
 
 
def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
 
 
def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #df = pd.read_csv(config.data_path)  # 填你自己的数据地址
    df = config.data_df.copy() #浅复制，保护原数据
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口
 
    cols_data = df.columns[1:] #获取index，切掉Date
    #cols_data = df.columns[0:] #获取index
    df_data = df[cols_data]
 
    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values
 
    # 定义标准化优化器
    scaler_train = StandardScaler()
    print("训练集尺寸:", len(true_data))

    true_data_list = []
    for i in range(config.input_size):
        init = 1  # 设初始因子值为1
        list_temp = []
        for j in range(len(true_data)):
            factor = (1 + true_data[j, i]) * init
            init = factor
            list_temp.append(init)
        true_data_list.append(list_temp)
    true_data_fac = np.array(true_data_list).transpose()
    factor_df = pd.DataFrame(true_data_fac)

    # 进行标准化处理
    true_data_normalized = scaler_train.fit_transform(true_data)

    #同样对因子值进行标准化处理
    true_data_fac_normalized = scaler_train.fit_transform(true_data_fac)

    # 转化为深度学习模型需要的类型Tensor
    true_data_normalized = torch.FloatTensor(true_data_normalized).to(device)

    # 因子值转化为深度学习模型需要的类型Tensor
    true_data_fac_normalized = torch.FloatTensor(true_data_fac_normalized).to(device)

    # 定义训练器的的输入
    true_inout_seq = create_inout_sequences(true_data_normalized, train_window, pre_len, config)

    # 定义因子训练器的的输入
    true_fac_inout_seq = create_inout_sequences(true_data_fac_normalized, train_window, pre_len, config)

    # 创建数据集
    true_dataset = TimeSeriesDataset(true_inout_seq)

    # 创建因子数据集
    true_fac_dataset = TimeSeriesDataset(true_fac_inout_seq)
 
    # 创建 DataLoader
    train_loader = DataLoader(true_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # 创建 因子DataLoader
    train_fac_loader = DataLoader(true_fac_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
 
    print("通过滑动窗口共有训练集数据：", len(true_inout_seq), "转化为批次数据:", len(train_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, scaler_train, train_fac_loader, df_data, factor_df

#加入自注意力机制
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

#long short term memory model
class TPALSTM(nn.Module):
    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers, device):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bias=True, batch_first=True)  # output (batch_size, obs_len, hidden_size)
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
 
        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(self.device)  # (batch_size, obs_len-1, hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)  # (num_layers, batch_size, hidden_size)
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

def lossFunctionFactor(pred_tensor, real_tensor):
    #输入预测值和真值的tensor序列，计算他们的Loss，之后返回一个tensor格式的Loss
    device = torch.device("cpu")

    pred_data = pred_tensor.detach().cpu().numpy()
    real_data = real_tensor.detach().cpu().numpy()

    pred_data_list = []
    for i in range(pred_data.shape[1]):
        init = 1  # 设初始因子值为1
        list_temp = []
        for j in range(len(pred_data)):
            factor = (1 + pred_data[j, i]) * init
            init = factor
            list_temp.append(init)
        pred_data_list.append(list_temp)
    pred_data_fac = np.array(pred_data_list).transpose()

    real_data_list = []
    for i in range(real_data.shape[1]):
        init = 1  # 设初始因子值为1
        list_temp = []
        for j in range(len(real_data)):
            factor = (1 + real_data[j, i]) * init
            init = factor
            list_temp.append(init)
        real_data_list.append(list_temp)
    real_data_fac = np.array(real_data_list).transpose()

    pred_tensor = torch.FloatTensor(pred_data_fac).to(device).requires_grad_(True)
    real_tensor = torch.FloatTensor(real_data_fac).to(device).requires_grad_(True)
    loss_function = nn.MSELoss()
    loss_tensor = loss_function(pred_tensor, real_tensor)
    return loss_tensor


def train(model, args, device, train_loader):
    start_time = time.time()  # 计算起始时间,用于展示进度
    lstm_model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005) #设定优化器和学习率
    epochs = args.epochs
    lstm_model.train()  # 训练模式
    results_loss = []
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad() #将上一个batch获得的梯度清零
            lstm_model.train()
 
            optimizer.zero_grad()
 
            y_pred = lstm_model(seq)

            single_loss = loss_function(y_pred, labels)  # type(single_loss) = tensor

            single_loss.backward()
 
            optimizer.step() #执行一次优化步骤，使用优化器更新参数值
            losss.append(single_loss.detach().cpu().numpy()) #返回与计算图分离且无梯度的新张量，并且将其转化为array
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss)) #返回loss的平均值
        torch.save(lstm_model.state_dict(), 'save_model.pth')#将模型储存在目录下
        time.sleep(0.1) #程序暂停0.1s

def predict(model, args, device, scaler, data):
    # 预测未知数据的功能
    # 重新读取数据
    df = data.copy() #浅复制，保护原数据
    #df = args.data_df
    #df = df.iloc[:, 1:][-args.window_size:].values  # 读取最后一个窗口数据，并转换为nadarry,切除第一列时间“Date”
    df = df.iloc[:, 0:][-args.window_size:].values  # 读取最后一个窗口数据，并转换为nadarry
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)   # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()  # 评估模式
 
    pred = model(tensor_pred)[0]

    if args.feature == 'M' or args.feature == 'S':
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    pred_df = pd.DataFrame(pred)
    
    return pred_df

def getFactorRate(factor_df, pred_df, config):
    full_data = pd.concat([factor_df, pred_df], axis=0)
    pred_rate = full_data.pct_change(1)
    pred_rate = pred_rate.tail(n = config.pre_len)
    return pred_rate

def LSTMargs(data_df, method = 'rate', window_size = 60, pre_len = 15, size = 8, lr = 0.001, drop_out = 0.05, epochs = 50, batch_size = 16):
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
    parser.add_argument('-window_size', type=int, default = window_size, help="时间窗口大小, window_size > pre_len") #可能需要调整
    parser.add_argument('-pre_len', type=int, default = pre_len, help="预测未来数据长度") #可能需要调整
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help="是否打乱数据加载器中的数据顺序")
    #parser.add_argument('-data_path', type=str, default = data_path, help="你的数据地址") #可能需要调整
    parser.add_argument('-data_df', type=str, default = data_df, help="你的数据") #可能需要调整
    parser.add_argument('-input_size', type=int, default = size, help='你的特征个数不算时间那一列') #可能需要调整
    parser.add_argument('-output_size', type=int, default = size, 
                        help='输出特征个数只有两种选择和你的输入特征一样即输入多少输出多少，另一种就是多元预测单元')#可能需要调整
    
    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    
    # learning 模型具体参数调整
    parser.add_argument('-lr', type=float, default=lr, help="学习率")
    parser.add_argument('-drop_out', type=float, default=drop_out, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=epochs, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=batch_size, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')
    parser.add_argument('-method', type=str, default= method, help = "loss计算方式") #可能需要调整
 
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
    args_LSTM = parser.parse_args()

    return args_LSTM

def pred(args):
    device = torch.device("cpu")
    print(device)
    train_loader, scaler_train, train_fac_loader, rate_df, factor_df = create_dataloader(args, device)
 
    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = TPALSTM(args.input_size,args.output_size,args.hidden_size, args.pre_len, args.laryer_num, device).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")
 
 
    # 训练模型
    if args.method == 'rate':
        if args.train:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            train(model, args, device, train_loader)
        if args.predict:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            pred = predict(model, args, device, scaler_train, rate_df)
        return pred
    else:
        if args.train:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            train(model, args, device, train_fac_loader)
        if args.predict:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            pred = predict(model, args, device, scaler_train, factor_df)
            pred_rate = getFactorRate(factor_df, pred, args)
        return pred_rate


if __name__ == '__main__':
    data = pd.read_csv('asset.csv')
    args1 = LSTMargs(data, epochs = 30)
    pred_val = pred(args1)
    pred_val.to_csv('pred_rate.csv')
    args2 = LSTMargs(data, epochs=30, method='factor')
    pred_fac = pred(args2)
    pred_fac.to_csv('pred_fac.csv')



