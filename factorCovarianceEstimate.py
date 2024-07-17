###import packages###
from __future__ import division
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import gamma
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS

#进行滑动窗口误差检验。与收益率预测采取同等大小的窗口，及使用60个交易日的数据进行计算，再用接下来15个交易日的数据进行评估。
#评估得分标准为协方差阵估计量与真实协方差阵的余弦相似度。以此标准，最终选择计算协方差使用的方法。

###划分train_set以及test_set###
def createXY(data, n_past = 60, n_future = 15):
    # dataset = pd.read_csv(data,parse_dates=["Date"],index_col=[0])
    dataset = data.copy()
    # dataset.set_index("Date", inplace=True)
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)-n_future):
        dataX.append(dataset[i - n_past:i])
        dataY.append(dataset[i:i + n_future])
    return np.array(dataX),np.array(dataY)

###Hilbert-Schmidt independence criterion计算方式###
def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2* np.dot(pattern1, pattern2.T)
    H = np.exp(-H/2/(deg**2))
    return H


def hsic_gam(X, Y, alph = 0.5):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.shape[0]

    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )

    # ----- width of X -----
    Ymed = Y

    G = np.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    bone = np.ones((n, 1), dtype = float)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6)**2

    varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

    varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC

    thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

    return (testStat, thresh)

def cov_estimator(train, test):
    method_list = ['经验协方差','收缩协方差','Ledoit-Wolf协方差','Orcale近似收缩协方差']#如果增加方法，则在此列表中增加名称
    score_array = [] #记录不同判断方法得分的列表
    cov_record = [] #记录最后一个窗口协方差的列表
    for i in range(0,train.shape[0]):
        #切片并转化为df格式
        train_win = pd.DataFrame(train[i])
        test_win = pd.DataFrame(test[i])
        
        cov_list = []
        #计算不同估计方法下的协方差阵并记录
        ###Emprirical Covariance###
        from sklearn.covariance import EmpiricalCovariance
        object = EmpiricalCovariance(store_precision=True,assume_centered=False)
        cov = object.fit(train_win) #返回一个covariance object
        cov_list.append(cov.covariance_) #返回拟合的经验协方差矩阵    
    
        ###Shrunk Covariance###    
        from sklearn.covariance import ShrunkCovariance
        object = ShrunkCovariance(store_precision=True,assume_centered=False,shrinkage=0.1)
        object.fit(train_win) #训练数据
        cov_list.append(object.covariance_) #给出拟合的收缩协方差矩阵    

        ###Ledoit Wolf Covariance###
        from sklearn.covariance import LedoitWolf
        object = LedoitWolf(store_precision=True,assume_centered=False,block_size=1000)
        object.fit(train_win)
        cov_list.append(object.covariance_) #获得收缩协方差矩阵
        
        ###Orcale 近似收缩###
        from sklearn.covariance import OAS
        object = OAS(store_precision=True,assume_centered=False)
        object.fit(train_win)
        cov_list.append(object.covariance_) #返回shrunk covariance matrix
        
        if i == train.shape[0]-1:
            cov_record = cov_list #将最后一期的协方差值保存在列表中
    
        #以测试集的协方差值作为真值进行评价
        real_value = np.array(test_win.cov())
        #记录不同计算方法的HSIC值，并进行比较
        score = []
        temp_cov = []
        for t in range(0,len(cov_list)):
            temp_cov = cov_list[t]
            temp_cov_array = np.array(temp_cov)
            r,trash = hsic_gam(real_value,temp_cov_array)
            score.append(r)
        score_array.append(score)#记录下本窗口的score值
    
    score_df = pd.DataFrame(score_array,columns = method_list)#转换为dataframe，并且命名
    best_method = score_df.mean().idxmax()#最佳方法为平均余弦值最大的方法
    cov_index = score_df.mean().argmax() #获得平均值得分最大的协方差估计方法的索引值
    cov_df = pd.DataFrame(cov_record[cov_index]) #取得最大得分方法的最后一个窗口的协方差估计值
    print(">>>>>"+best_method+"<<<<<") #输出最优方法
    return cov_df

if __name__=="__main__":
    ###import data&divide into train&test set###
    position = "data.csv" #如果不在同一目录下，需要使用绝对位置
    #N_past = 60,N_future = 15,using 60 days past to predict the 61-75 value
    #选择使用历史数据窗口时间和预测数据窗口时间
    trainX,testX=createXY(position,60,15)

    #获得评判标准下，不同判断方法的得分（滑动平均值）#
    cov_estimator(trainX,testX)