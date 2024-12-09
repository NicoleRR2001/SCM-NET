import torch
import numpy as np
from toolbox import myDataset_5cv

'''
file name: DE_4D_Feature
input: 1. choose nth fold of 5-fold
       2. batch_size
       3. random seed
       4. the 4D feature of all subjects
       
output: train_dataloader and test_dataloader of nth fold
'''
#torch.Size([20355, 16, 5, 6, 9])
#16：twice number of segments;电极地图的长和宽为6*9，5个频率带；20355*16 总数据量大小
#torch.Size([20355, 1])
"""
EEG 数据集17个电极，五个频段，电极地图大小是6*9
2T=16 twice number of segments
"""

#  choose nth fold63
n=2
# batch_size
batch_size = 256
# batch_size = 150

# 随机数种子
seed = 20
# 使用gpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### 1.数据集 ###
data = np.load('./processedData/data_4d.npy')   # 20355
label = np.load('./processedData/label.npy')


#转为32浮点型
data = torch.FloatTensor(data)
label = torch.FloatTensor(label)

print(data.shape)
print(label.shape)
print(n)

# 五折交叉验证
train_dataloader, test_dataloader = myDataset_5cv(data, label, batch_size, n, seed)
# print(train_dataloader.second)
