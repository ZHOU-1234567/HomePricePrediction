# 忽略警告
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, skew      # 获取统计信息
from scipy import stats

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))    # 限制浮点数输出为小数点后三位

# 加载数据集
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# 检查样本和variables的数量
print("删除ID列前的train大小：{}".format(data_train.shape))
print("删除ID列前的test大小：{}".format(data_test.shape))

# 保存ID列
train_ID = data_train['Id']
test_ID = data_test['Id']

# 删除ID列，因为在数据分析中ID列是用不上的
data_train.drop("Id", axis=1, inplace=True)
data_test.drop("Id", axis=1, inplace=True)

""" 
下面是异常值处理，目标变量分析，缺失值处理，特征相关性
进一步挖掘特征，对特征进行box-cox变换，独热编码
"""

"""
# 首先通过绘制散点图，直观看出train数据是否有离群值
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.scatter(x=data_train['GrLivArea'], y=data_train['SalePrice'])
plt.ylabel('SalePrice',fontsize=13)
plt.xlabel('GrLivArea',fontsize=13)
"""

# 把离群值去除掉
data_train = data_train.drop(data_train[(data_train['GrLivArea'] > 4000) & (data_train['SalePrice'] < 300000)].index)

"""
plt.subplot(122)
plt.scatter(data_train['GrLivArea'],data_train['SalePrice'])
plt.ylabel('SalePrice',fontsize=13)
plt.xlabel('GrLivArea',fontsize=13)
"""

# 对SalePrice进行目标分析和处理

# 画出分布图
fig, ax = plt.subplots(nrows=2, figsize=(6,10))
sns.distplot(data_train['SalePrice'], fit=norm, ax=ax[0])
(mu, sigma) = norm.fit(data_train['SalePrice'])
ax[0].legend(['Normal dist. ($/mu=$ {:.2f} and $/sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
ax[0].set_ylabel('Frequency')
ax[0].set_title('SalePrice Distribution')
stats.probplot(data_train['SalePrice'], plot=ax[1])
plt.show()