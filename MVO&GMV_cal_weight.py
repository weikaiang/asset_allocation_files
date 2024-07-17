import pandas as pd
import numpy as np
import scipy.optimize as sco

# 该函数返回了夏普比率的负值，换句话说，我们需要对该函数进行最小化操作。之所以使用这种形式，是因为scipy.optimization只有minimize函数（机器学习中优化目标通常是最小化损失函数），所以这里使用了这种看起来有点奇怪的形式。
def min_func_sharpe(weights):
    """获取投资组合的各种统计值
参数:
        weights：分配的资金权重

返回值：
        p_ret：投资组合的收益率
        p_vol：投资组合的波动率
        p_sr：投资组合的夏普比率

    """
    w = np.array(weights)
    p_ret = np.sum(rets.mean()*w)*252
    p_vol = np.sqrt(np.dot(w.T,np.dot(covs*252,w)))
    p_sr = p_ret/p_vol
    
    """
    优化的目标函数，最小化夏普比率的负值，即最大化夏普比率
    """
    return -p_sr

def mvo_weight(rets, covs):
    n = len(rets.columns)
    cons = (
    {'type': 'eq',
     'fun': lambda x:np.sum(x)-1 #资金的权重和为1,sunw = 1。
    },
 
    {'type': 'ineq',
     'fun': lambda x: 0.075/252-np.dot(x.T,np.dot(covs*252,x)) #跟踪误差控制在年化7.5%以内,即：7.5%>w'(∑est)w.
    }
    #还可以添加其他条件，使得规划条件和策略要求相一致
    )
     #参数取值范围
    bnds = tuple((0,1) for x in range(n)) #权重非负（不能买空）
    
    # 生成初始权重
    w_initial = n*[1./n,]
    # 计算最优权重
    opts_sharpe = sco.minimize(min_func_sharpe,w_initial,method='SLSQP',bounds=bnds,constraints=cons)
    # fun对应最优夏普比率,注意取负值，x为权重。
    mvo_weight = opts_sharpe['x'].round(4)
    return mvo_weight

###第一步 先寻找得到最小方差边界（有效前沿）和最小方差组合（GMVP）###
def annualize_rets(returns,n_periods):
    '''
    给定一系列的收益率和期数，算出年化收益率
    '''
    # 每一期的平均收益
    r_periodic_mean = ((1+returns).prod())**(1/returns.shape[0])-1
    return (1+r_periodic_mean)**n_periods-1
 
def annualize_std(returns,n_periods):
    '''
    给定一系列的收益率，算出年化的标准差
    '''
    return returns.std()*np.sqrt(n_periods)
 
def portfolio_return(weights,returns):
    '''
    计算因子组合收益率，weights和returns需要矩阵形式
    weights是组合因子的权重
    returns是组合中的因子年化收益率
    '''
    return weights.T @ returns
 
def portfolio_vol(weights,covmat):
    '''
    计算因子组合风险（波动率），weights和covmat需要矩阵形式
    covmat代表的是协方差矩阵
    '''
    return np.sqrt(weights.T @ covmat @ weights)

# GMVP: Global Min Variance Portfolio
def get_gmvp(covmat):
    '''
    寻找全局最小方差点
    covmat 代表因子之间的协方差矩阵
    '''
    from scipy.optimize import minimize
    n = covmat.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n #每个因子的权重在0~1之间
    weights_sum_to_1 = {'type':'eq','fun': lambda weights:np.sum(weights)-1} #不同因子的权重和为1
    weights = minimize(portfolio_vol,init_guess,args=(covmat,),
                       method='SLSQP',bounds=bounds,constraints=(weights_sum_to_1))
    return weights.x

if __name__ == '__main__':
    # 投资组合
    data = pd.read_csv("data.csv",parse_dates=["Date"],index_col=[0])
    rets = data #导入因子收益率数据(格式为dataframe)
    covs = data.cov() #导入协方差阵(格式为dataframe)
    mvo_weight = mvo_weight(rets, covs)
    gmvp_weight = get_gmvp(covs)