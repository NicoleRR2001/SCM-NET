import os.path
from myNet import *
from AMS import ams_cnn
from EEGResNet import EEGResNet18
from EEGNet import EEGNet
from ESTCNN import *
from toolbox import *
from visdom import Visdom
from dataloader import *
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sklearn import datasets

import torchvision
# from thop import profile


warnings.filterwarnings("ignore")

'''
file name: DE_4D_Feature
input: 1. epoch
       2. optimizer

output: model.pth

Warning!!!!!
Before train begin, start up the visdom to Real-time monitoring accuracy
'''


#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
myModel = SFT_Net()
# myModel = ESTCNN()
# myModel = ams_cnn()
#myModel=EEGResNet18()
# myModel=EEGNet()
netname=("seed-vig-MAMBA"
         "")
# netname="EEGNet"
# netname="ams_cnn"
#netname="seed_vig_mamba500"
print(netname)

classes=['Awake','Fatigue']
local_name=time.strftime('%Y-%m-%d-%H-%M-%S')
name = str(local_name) +str(netname)+ "_train_acc.csv"
df = pd.DataFrame(columns=['train step','train average Loss','train average accuracy','test average Loss','test average accuracy','test average pre','test average recall','test average kappa','test average f1','confusion_mat'])#列名
df.to_csv(name,index=False) #路径可以根据需要更改

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # cuda 指定使用GPU设备
# model = torch.nn.DataParallel(myModel, device_ids=[1])  # 指定多GPU并行处理时使用的设备编号
# # save highest accuracy model.pth
acc_low = 0.8
acc_list = []

myModel = myModel.to(device)

device = torch.device("cuda:1")
myModel.to(device)
# loss function
loss_fn = torch.nn.MSELoss()
# optimizer
learningRate = 2e-3
#weight_decay=0.02 表示 L2 正则化项的系数
optimizer = torch.optim.AdamW(myModel.parameters(), lr=learningRate, weight_decay=0.02) # AdamW

epoch = 500
# Record total step and loss
total_train_step = 0
total_test_step = 0
total_train_loss = 0
total_test_loss = 0
total_test_pre=0
total_test_recall=0
total_test_kappa=0

total_test_f1=0
# Visdom
viz = Visdom()
train_loss_viz = 0
test_loss_viz = 0
acc_viz = 0
total_train_viz = 0
total_test_viz = 0
# Visdom init
viz.line([[1, 1]], [0], win='loss', opts=dict(title='loss', legend=['train', 'test']))
viz.line([[0, 0]], [0], win='acc', opts=dict(title='acc', legend=['train', 'test']))



for i in range(epoch):
    print("--------------The {}th epoch of training starts------------".format(i + 1))
    total_train_loss = 0
    total_train_acc = 0
    ### 1.TRAIN ###
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    for data in train_dataloader:
        x, y = data
        # print(x,y)
        x = x.to(device)
        y = y.to(device)
        # ignore attention output when training
        # outputs, _sa, _fa = myModel(x)
        train_data,outputs_train,ssconv_x11, ssconv_x12,ssconv_x21, ssconv_x22,ssconv_x31, ssconv_x32,ssconv_x41, ssconv_x42,ssconv_x51, ssconv_x52,ssconv_x61, ssconv_x62,ssconv_x71, ssconv_x72,ssconv_x81, ssconv_x82,ssconv_x91, ssconv_x92,ssconv_x101, ssconv_x102,ssconv_x111, ssconv_x112,ssconv_x121, ssconv_x122,ssconv_x131, ssconv_x132,ssconv_x141, ssconv_x142,ssconv_x151, ssconv_x152,ssconv_x161, ssconv_x162= myModel(x)
        train_loss = loss_fn(outputs_train, y)
        # Calculate accuracy by toolbox.label_2class
        label_train = label_2class(y)
        label_train_pred = label_2class(outputs_train)
        total_train_acc += accuracy_score(label_train, label_train_pred)
        # Gradient update 更新梯度，反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Calculate loss
        total_train_loss = total_train_loss + train_loss.item()
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("train step：{}，train average loss：{:.6f}".format(total_train_step, total_train_loss/total_train_step))
    total_train_viz += 1
    # in this epoch, train accuracy
    print("train average accuracy: {:.4f}%".format(100.0 * (total_train_acc/len(train_dataloader))))

    ### 2.TEST ###
    total_test_acc = 0
    total_test_loss = 0
    r2 = 0
    attentionGraph = torch.Tensor()
    with torch.no_grad():
        for data in test_dataloader:
            testx, testy = data
            testx = testx.to(device)
            testy = testy.to(device)
            # outputs, spaAtten, freqAtten = myModel(testx)
            a,outputs,x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2,x13_1,x13_2,x14_1,x14_2,x15_1,x15_2,x16_1,x16_2= myModel(testx)
            label = label_2class(testy)
            label_pred = label_2class(outputs)
            test_loss = loss_fn(outputs, testy)
            total_test_loss = total_test_loss + test_loss.item()
            total_test_step = total_test_step + 1
            # use toolbox.myEvaluate to calculate
            conf, acc, report, pre, recall, f1, kappa = myEvaluate(label, label_pred)
            # total accuracy
            total_test_acc += acc
            test_pre = precision_score(label, label_pred, average='macro')
            test_recall = recall_score(label, label_pred, average='macro')
            test_f1 = f1_score(label, label_pred, average='macro')
            test_kappa = cohen_kappa_score(label, label_pred)



    total_test_viz += 1
    print("test average loss:{:.6f}".format(total_test_loss / total_test_step))
    # in this epoch test accuracy
    print("test average accuracy:{:.4f}%".format(100.0 * (total_test_acc / len(test_dataloader))))
    print("test average pre:{:.4f}".format(test_pre))
    print("test average recall:{:.4f}".format(test_recall))
    print("test average kappa:{:.4f}".format(test_kappa))
    print("test average f1:{:.4f}".format(test_f1))
    # 使用sklearn工具中confusion_matrix方法计算混淆矩阵
    confusion_mat = confusion_matrix(label_train, label_train_pred)
    print("confusion_mat.shape : {}".format(confusion_mat.shape))
    print("confusion_mat : {}".format(confusion_mat))
    # print(last_data_train)
    print("===")
    print(label_train)

    # 保存训练数据
    list=[total_train_step,total_train_loss/total_train_step,100.0 * (total_train_acc/len(train_dataloader)),total_test_loss / total_test_step,100.0 * (total_test_acc / len(test_dataloader)),test_pre,test_recall,test_kappa,test_f1,confusion_mat]
    data = pd.DataFrame([list])
    data.to_csv(name, mode='a', header=False, index=False)


    # viz.line([[total_train_loss/len(train_dataloader), total_test_loss/len(test_dataloader)]], [i+1], win='loss', update="append")
    # viz.line([[total_train_acc/len(train_dataloader), total_test_acc/len(test_dataloader)]], [i+1], win='acc', update="append")

    #  If test accuracy more than the acc_low, save the model.pth
    if (total_test_acc/len(test_dataloader)) > acc_low:
        acc_low = (total_test_acc/len(test_dataloader))
        torch.save(myModel.state_dict(), './pth/model_MAMBA_fold_%d.pth' % n)

    train_data=train_data.detach()
    train_data=train_data.cpu()
    train_data=train_data.numpy()
    #
    # ssconv_x1 = ssconv_x1.detach()
    # ssconv_x1 = ssconv_x1.cpu()
    # ssconv_x1 = ssconv_x1.numpy()
    #
    # ssconv_x2 = ssconv_x2.detach()
    # ssconv_x2 = ssconv_x2.cpu()
    # ssconv_x2 = ssconv_x2.numpy()
    # for i in range(len(ssconv)):

    print(ssconv_x11.shape)
    batch_size=ssconv_x21.size(0)
    # print(ssconv_x1.shape)
    ssconv_x11 = ssconv_x11.detach()
    ssconv_x11 = ssconv_x11.cpu()
    ssconv_x11 = ssconv_x11.numpy()
    ssconv_x12 = ssconv_x12.detach()
    ssconv_x12 = ssconv_x12.cpu()
    ssconv_x12 = ssconv_x12.numpy()

    ssconv_x21 = ssconv_x21.detach()
    ssconv_x21 = ssconv_x21.cpu()
    ssconv_x21 = ssconv_x21.numpy()
    ssconv_x22 = ssconv_x22.detach()
    ssconv_x22 = ssconv_x22.cpu()
    ssconv_x22 = ssconv_x22.numpy()

    ssconv_x31 = ssconv_x31.detach()
    ssconv_x31 = ssconv_x31.cpu()
    ssconv_x31 = ssconv_x31.numpy()
    ssconv_x32 = ssconv_x32.detach()
    ssconv_x32 = ssconv_x32.cpu()
    ssconv_x32 = ssconv_x32.numpy()

    ssconv_x41 = ssconv_x41.detach()
    ssconv_x41 = ssconv_x41.cpu()
    ssconv_x41 = ssconv_x41.numpy()
    ssconv_x42 = ssconv_x42.detach()
    ssconv_x42 = ssconv_x42.cpu()
    ssconv_x42 = ssconv_x42.numpy()

    ssconv_x51 = ssconv_x51.detach()
    ssconv_x51 = ssconv_x51.cpu()
    ssconv_x51 = ssconv_x51.numpy()
    ssconv_x52 = ssconv_x52.detach()
    ssconv_x52 = ssconv_x52.cpu()
    ssconv_x52 = ssconv_x52.numpy()

    ssconv_x61 = ssconv_x61.detach()
    ssconv_x61 = ssconv_x61.cpu()
    ssconv_x61 = ssconv_x61.numpy()
    ssconv_x62 = ssconv_x62.detach()
    ssconv_x62 = ssconv_x62.cpu()
    ssconv_x62 = ssconv_x62.numpy()

    ssconv_x71 = ssconv_x71.detach()
    ssconv_x71 = ssconv_x71.cpu()
    ssconv_x71 = ssconv_x71.numpy()
    ssconv_x72 = ssconv_x72.detach()
    ssconv_x72 = ssconv_x72.cpu()
    ssconv_x72 = ssconv_x72.numpy()

    ssconv_x81 = ssconv_x81.detach()
    ssconv_x81 = ssconv_x81.cpu()
    ssconv_x81 = ssconv_x81.numpy()
    ssconv_x82 = ssconv_x82.detach()
    ssconv_x82 = ssconv_x82.cpu()
    ssconv_x82 = ssconv_x82.numpy()

    ssconv_x91 = ssconv_x91.detach()
    ssconv_x91 = ssconv_x91.cpu()
    ssconv_x91 = ssconv_x91.numpy()
    ssconv_x92 = ssconv_x92.detach()
    ssconv_x92 = ssconv_x92.cpu()
    ssconv_x92 = ssconv_x92.numpy()

    ssconv_x101 = ssconv_x101.detach()
    ssconv_x101 = ssconv_x101.cpu()
    ssconv_x101 = ssconv_x101.numpy()
    ssconv_x102 = ssconv_x102.detach()
    ssconv_x102 = ssconv_x102.cpu()
    ssconv_x102 = ssconv_x102.numpy()

    ssconv_x111 = ssconv_x111.detach()
    ssconv_x111 = ssconv_x111.cpu()
    ssconv_x111 = ssconv_x111.numpy()
    ssconv_x112 = ssconv_x112.detach()
    ssconv_x112 = ssconv_x112.cpu()
    ssconv_x112 = ssconv_x112.numpy()

    ssconv_x121 = ssconv_x121.detach()
    ssconv_x121 = ssconv_x121.cpu()
    ssconv_x121 = ssconv_x121.numpy()
    ssconv_x122 = ssconv_x122.detach()
    ssconv_x122 = ssconv_x122.cpu()
    ssconv_x122 = ssconv_x122.numpy()

    ssconv_x131 = ssconv_x131.detach()
    ssconv_x131 = ssconv_x131.cpu()
    ssconv_x131 = ssconv_x131.numpy()
    ssconv_x132 = ssconv_x132.detach()
    ssconv_x132 = ssconv_x132.cpu()
    ssconv_x132 = ssconv_x132.numpy()

    ssconv_x141 = ssconv_x141.detach()
    ssconv_x141 = ssconv_x141.cpu()
    ssconv_x141 = ssconv_x141.numpy()
    ssconv_x142 = ssconv_x142.detach()
    ssconv_x142 = ssconv_x142.cpu()
    ssconv_x142 = ssconv_x142.numpy()

    ssconv_x151 = ssconv_x151.detach()
    ssconv_x151 = ssconv_x151.cpu()
    ssconv_x151 = ssconv_x151.numpy()
    ssconv_x152 = ssconv_x152.detach()
    ssconv_x152 = ssconv_x152.cpu()
    ssconv_x152 = ssconv_x152.numpy()

    ssconv_x161 = ssconv_x161.detach()
    ssconv_x161 = ssconv_x161.cpu()
    ssconv_x161 = ssconv_x161.numpy()
    ssconv_x162 = ssconv_x162.detach()
    ssconv_x162 = ssconv_x162.cpu()
    ssconv_x162 = ssconv_x162.numpy()

    data11 = ssconv_x11
    data12=ssconv_x12

    data21=ssconv_x21
    data22=ssconv_x22

    data31 = ssconv_x31
    data32 = ssconv_x32

    data41 = ssconv_x41
    data42 = ssconv_x42

    data51 = ssconv_x51
    data52 = ssconv_x52

    data61 = ssconv_x61
    data62 = ssconv_x62

    data71 = ssconv_x71
    data72 = ssconv_x72

    data81 = ssconv_x81
    data82 = ssconv_x82

    data91 = ssconv_x91
    data92 = ssconv_x92

    data101 = ssconv_x101
    data102 = ssconv_x102

    data111 = ssconv_x111
    data112 = ssconv_x112

    data121 = ssconv_x121
    data122 = ssconv_x122

    data131 = ssconv_x131
    data132 = ssconv_x132

    data141 = ssconv_x141
    data142 = ssconv_x142

    data151 = ssconv_x151
    data152 = ssconv_x152

    data161 = ssconv_x161
    data162 = ssconv_x162
    if i%30==0:
        data_list1 = []
        data_list2 = []
        num = 15
        for data in [data11, data21, data31, data41, data51, data61, data71, data81,
                     data91, data101, data111, data121, data131, data141, data151, data161]:
            data_list1.append(data[num])

        sample_data1 = np.stack(data_list1, axis=0)
        print(len(data_list1))
        print(sample_data1.shape)
        reshaped_data = []
        all_data = np.empty([0, 54])
        all_data2 = np.empty([0, 54])

        num_rows = 5
        num_cols = 1
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 50))
        for i in range(5):
            sample = sample_data1[:, i, :, :]
            reshaped_data = sample.reshape(16, -1)
            all_data = np.vstack([all_data, reshaped_data])
            plt.figure(figsize=(10, 8))
            sns.heatmap(reshaped_data, cmap="Blues", ax=axes[i])
            axes[i].set_title(f"Sample {num} Dimension {i + 1}")
            axes[i].set_xlabel("Feature Index")
            axes[i].set_ylabel(f"Channel {i + 1}")
            # sns.heatmap(reshaped_data, cmap="Blues")
            # # 在对应的子图中绘制热力图
            # plt.title(f"Sample {num} Dimension {i + 1}")
            # plt.xlabel("Feature Index")
            # plt.ylabel(f"Channle {i + 1}")
            # plt.show()
            # 调整子图布局，使整体更美观
        #
        # plt.tight_layout()
        # # 统一展示拼接好的图
        # plt.show()
        print(all_data.shape, "66666")
        sns.heatmap(all_data, cmap="Blues")
        np.save('./pth/seed_vig_heatmap_up.npy', all_data)
        # 在对应的子图中绘制热力图
        plt.title(f"Sample {num} Heatmap of Up")
        plt.xlabel("Feature Index")
        plt.ylabel(f"Channle ")
        plt.show()

        for data in [data12, data22, data32, data42, data52, data62, data72, data82,
                     data92, data102, data112, data122, data132, data142, data152, data162]:
            data_list2.append(data[num])

        sample_data2 = np.stack(data_list2, axis=0)
        print(len(data_list2))
        print(sample_data2.shape)
        reshaped_data = []

        num_rows = 1
        num_cols = 5
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
        for i in range(5):
            sample = sample_data2[:, i, :, :]
            print(sample.shape)
            reshaped_data = sample.reshape(16, -1)
            print(reshaped_data.shape)
            plt.figure(figsize=(10, 8))
            all_data2 = np.vstack([all_data2, reshaped_data])
            # RdBu
            sns.heatmap(reshaped_data, cmap="Blues")
            # 在对应的子图中绘制热力图
            plt.title(f"Sample {num} Dimension {i + 1}")
            plt.xlabel("Feature Index")
            plt.ylabel(f"Channle {i + 1}")
            plt.show()
        print(all_data2.shape, "66666")
        sns.heatmap(all_data2, cmap="Blues")
        np.save('./pth/seed_vig_heatmap_down.npy', all_data2)
        # 在对应的子图中绘制热力图
        plt.title(f"Sample {num} Heatmap of Down")
        plt.xlabel("Feature Index")
        plt.ylabel(f"Channle ")
        plt.show()
    #
    # data_list1=[]
    # data_list2 = []
    # num=15
    # for data in [data11, data21, data31, data41, data51, data61, data71, data81,
    #              data91, data101, data111, data121, data131, data141, data151, data161]:
    #     data_list1.append(data[num])

    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[8], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {8}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # #
    # # for i in range(len(data101)):
    # #     sample = data101[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[9], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {9}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # #
    # # for i in range(len(data111)):
    # #     sample = data111[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[10], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {10}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # #
    # # for i in range(len(data121)):
    # #     sample = data121[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[11], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {11}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # #
    # # for i in range(len(data131)):
    # #     sample = data131[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[12], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {12}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # #
    # # for i in range(len(data141)):
    # #     sample = data141[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[13], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {13}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # #
    # # for i in range(len(data151)):
    # #     sample = data151[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[14], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {14}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    # #
    # # for i in range(len(data161)):
    # #     sample = data161[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data11.append(sample_reshaped)
    # # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    # #     reshaped_data11.shape)
    # # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # # for i in range(143):
    # #     for j in range(5):
    # #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    # #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    # #
    # # print(new_data11.shape)
    # # # 一个batch一张
    # # # for i in range(10):
    # # band_data1.append(new_data11[5])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(band_data1[15], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of Sample {15}")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # reshaped_data11 = []
    #
    # band1=X = np.empty([143, 17, 0])
    # for i in range(len(data12)):
    #     sample = data12[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # reshaped_data11=reshaped_data11[reshaped_data11!=0]
    # # print(reshaped_data11.shape)
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape,"++++++",new_data11.dtype)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[0], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {5}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data21 = []
    #
    #
    #
    # for i in range(len(data22)):
    #     sample = data22[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data21.append(sample_reshaped)
    # reshaped_data21 = np.array(reshaped_data21)  # 将列表转换为numpy数组
    # print(reshaped_data21.shape)
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data21.reshape(-1, reshaped_data21.shape[2])).reshape(
    #     reshaped_data21.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    #
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[1], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {1}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data32)):
    #     sample = data32[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[2], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {2}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data42)):
    #     sample = data42[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[3], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {3}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data52)):
    #     sample = data52[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[4], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {4}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data62)):
    #     sample = data62[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[5], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {5}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data72)):
    #     sample = data72[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[6], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {6}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data82)):
    #     sample = data82[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[7], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {7}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data92)):
    #     sample = data92[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[8], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {8}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data102)):
    #     sample = data102[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[9], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {9}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data112)):
    #     sample = data112[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[10], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {10}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data122)):
    #     sample = data122[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[11], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {11}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data132)):
    #     sample = data132[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[12], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {12}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data142)):
    #     sample = data142[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[13], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {13}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # for i in range(len(data152)):
    #     sample = data152[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[14], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {14}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    # for i in range(len(data162)):
    #     sample = data162[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data11.append(sample_reshaped)
    # reshaped_data11 = np.array(reshaped_data11)  # 将列表转换为numpy数组
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data11.reshape(-1, reshaped_data11.shape[2])).reshape(
    #     reshaped_data11.shape)
    # new_data11 = np.zeros((143, 5, len(index1)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, index1]  # 提取指定索引位置的元素
    #         new_data11[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data11.shape)
    # # 一个batch一张
    # # for i in range(10):
    # band_data1.append(new_data11[5])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(band_data1[15], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {15}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    # reshaped_data11 = []
    #
    #
    # # plt.figure(figsize=(10, 8))
    # # print(len(band_data1))
    # # sns.heatmap(band_data1, cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # # plt.title(f"Heatmap of All")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    #
    #
    # # 方式二：合并绘制整体热力图（将所有样本数据合并后绘制一个热力图）
    # # 先将所有样本数据进一步展平合并成一个二维矩阵
    # # total_reshaped_data = reshaped_data_normalized.reshape(-1, reshaped_data_normalized.shape[2])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(total_reshaped_data, cmap="YlGnBu")
    # # plt.title("Heatmap of All Samples in down")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    #
    # #
    # #
    # # ssconv_x122 = ssconv_x122.detach()
    # # ssconv_x122 = ssconv_x122.cpu()
    # # ssconv_x122 = ssconv_x122.numpy()
    # data2= ssconv_x122
    #
    # reshaped_data2 = []
    # for i in range(len(data11)):
    #     sample = data2[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    #     reshaped_data2.append(sample_reshaped)
    # reshaped_data2 = np.array(reshaped_data2)  # 将列表转换为numpy数组
    # # print(reshaped_data2.shape)
    #
    #
    # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # scaler = MinMaxScaler()
    # reshaped_data_normalized = scaler.fit_transform(reshaped_data2.reshape(-1, reshaped_data2.shape[2])).reshape(
    #     reshaped_data2.shape)
    # print(reshaped_data_normalized.shape)
    # # print(reshaped_data_normalized)
    # #21，23，30，31，32，39，40，48，49
    # # new_data = np.concatenate(reshaped_data_normalized[4,4,21],reshaped_data_normalized[4,4,23],reshaped_data_normalized[4,4,30],reshaped_data_normalized[4,4,31],reshaped_data_normalized[4,4,32],reshaped_data_normalized[4,4,39],reshaped_data_normalized[4,4,40],reshaped_data_normalized[4,4,48],reshaped_data_normalized[4,4,49])
    # indices_to_extract = [21, 23, 30, 31, 32, 39, 40, 48, 49]
    #
    # new_data = np.zeros((143, 5, len(indices_to_extract)))  # 创建一个用于存储新数据的全零数组，形状为(143, 5, 9)
    #
    # for i in range(143):
    #     for j in range(5):
    #         selected_elements = reshaped_data_normalized[i, j, indices_to_extract]  # 提取指定索引位置的元素
    #         new_data[i, j] = selected_elements  # 将提取的元素赋值给新数据相应位置
    #
    # print(new_data.shape)
    # # 一个batch一张
    # # for i in range(len(reshaped_data_normalized)):
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(new_data[5], cmap="YlGnBu")  # 选择合适的颜色映射，这里用YlGnBu，你可按需更换
    # plt.title(f"Heatmap of Sample {5}")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Channel Index")
    # plt.show()
    #
    # #
    # # data2 = ssconv_x2
    # # reshaped_data2 = []
    # # for i in range(len(data2)):
    # #     sample = data2[i]  # 取出每个样本数据（对应batch_size维度中的每个元素）
    # #     sample_reshaped = sample.reshape(5, -1)  # 将每个样本数据展平成二维，并转为numpy数组
    # #     reshaped_data2.append(sample_reshaped)
    # # reshaped_data2 = np.array(reshaped_data2)  # 将列表转换为numpy数组
    # #
    # # # 数据归一化处理，这里使用MinMaxScaler将数据归一化到[0, 1]区间
    # # scaler = MinMaxScaler()
    # # reshaped_data_normalized = scaler.fit_transform(reshaped_data2.reshape(-1, reshaped_data2.shape[2])).reshape(
    # #     reshaped_data2.shape)
    # #
    # # # 方式二：合并绘制整体热力图（将所有样本数据合并后绘制一个热力图）
    # # # 先将所有样本数据进一步展平合并成一个二维矩阵
    # # total_reshaped_data = reshaped_data_normalized.reshape(-1, reshaped_data_normalized.shape[2])
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(total_reshaped_data, cmap="YlGnBu")
    # # plt.title("Heatmap of All Samples in down")
    # # plt.xlabel("Feature Index")
    # # plt.ylabel("Channel Index")
    # # plt.show()
    # # for batch_idx in range(batch_size):
    # #     batch_data = data1[batch_idx]  # 取出每个batch的数据
    # #     per_batch_reshaped = []
    # #     print(batch_data.shape)
    # #     for frame_idx in range(batch_data.size(0)):  # 这里的size(0)为16，对应16个数据块（可理解为类似图像的帧等情况）
    # #         frame = batch_data[frame_idx]
    # #         channel_reshaped = []
    # #         print(frame.shape)
    # #         for channel in range(frame.size(0)):  # 这里的size(0)为5，对应通道维度
    # #             channel_data = frame[channel].reshape(1, -1)  # 将每个通道数据展平成二维（1行，其他维度合并成一列）
    # #             channel_reshaped.append(channel_data)
    # #             # print(channel_data.shape)
    # #         per_frame_reshaped = torch.cat(channel_reshaped, dim=0)  # 将一个帧内各通道数据按行拼接，形成一个二维矩阵（通道数行，合并列数列）
    # #         per_batch_reshaped.append(per_frame_reshaped)
    # #         print(per_frame_reshaped.shape)
    # #     batch_reshaped = torch.cat(per_batch_reshaped, dim=1)  # 将一个batch内各帧的数据按列拼接
    # #     reshaped_data.append(batch_reshaped)
    # # reshaped_data = torch.cat(reshaped_data, dim=0)  # 将所有batch的数据按行拼接，最终形成二维矩阵
    # #
    # # print(reshaped_data.shape)
    # #
    # # # 如果batch_size大于1，并且想对每个样本单独绘制热力图，可以遍历batch_size维度进行后续操作
    # # if batch_size > 1:
    # #     for i in range(batch_size):
    # #         sample_data = reshaped_data[i * (16 * 5): (i + 1) * (16 * 5)]  # 取出每个batch对应的二维数据部分
    # #         plt.figure(figsize=(10, 8))
    # #         sns.heatmap(sample_data.detach().numpy(), cmap="YlGnBu")
    # #         plt.title(f"Heatmap of Batch {i}")
    # #         plt.xlabel("Feature Index")
    # #         plt.ylabel("Channel Index")
    # #         plt.show()
    # #
    # # # 如果batch_size等于1，直接基于reshaped_data绘制热力图
    # # else:
    # #     plt.figure(figsize=(10, 8))
    # #     sns.heatmap(reshaped_data.detach().numpy(), cmap="YlGnBu")
    # #     plt.title("Heatmap of Data")
    # #     plt.xlabel("Feature Index")
    # #     plt.ylabel("Channel Index")
    #     plt.show()

    # ssconv=ssconv.detach()
    # ssconv = ssconv.cpu()
    # ssconv = ssconv.numpy()
    #
    # ssconv_cpu = [tensor.cpu() for tensor in ssconv]
    # ssconv = np.array(ssconv_cpu)
    # ssconv = ssconv.cpu()
    # ssconv = ssconv.numpy()

    # label_train=label_train.detach()
    # label_train=label_train.cpu()
    # label_train=label_train.numpy()
    # # np.save('./pth/train_last_data.npy',last_data_train)
    # # np.save('./pth/train_data.npy', label_train)
    # np.save('./pth/MAMBA_data.npy', train_data)
    # np.save('./pth/MAMBA_label.npy', label_train)
    # np.save('./pth/ssconv_data1.npy',ssconv_x1)
    # np.save('./pth/ssconv_data2.npy', ssconv_x2)
    # # df = pd.DataFrame(label_pred,
    #                   index=[chr(i) for i in range(65, 90)],  # DataFrame的行标签设置为大写字母
    #                   columns=["a", "b", "c", "d", "e"])  # 设置DataFrame的列标签
    # plt.figure(dpi=120)
    # sns.heatmap(data=df,
    #             cmap=sns.diverging_palette(10, 220, sep=80, n=7),  # 区分度显著色盘：sns.diverging_palette()使用
    #             )
    # plt.title("使用seaborn diverging颜色盘：sns.diverging_palette(10, 220, sep=80, n=7)")





print("This is ", n, " fold, highest accuracy is: ", acc_low)

# ['train step','train average Loss','train average accuracy','test average Loss','test average accuracy']
# 读取csv中指定列的数据
data = pd.read_csv(name)
data_train_loss = data[['train average Loss']]  # class 'pandas.core.frame.DataFrame'
data_train_acc = data[['train average accuracy']]

data_test_loss = data[['test average Loss']]  # class 'pandas.core.frame.DataFrame'
data_test_acc = data[['test average accuracy']]

x = np.arange(0, epoch, 1)
y1 = np.array(data_train_loss)  # 将DataFrame类型转化为numpy数组
y2 = np.array(data_train_acc)

y3 = np.array(data_test_loss)  # 将DataFrame类型转化为numpy数组
y4 = np.array(data_test_acc)

#matplotlib.use('TkAgg')
local=time.strftime('%Y-%m-%d-%H-%M-%S')
new_name1 = str(local) +str(netname)+ "_loss.png"
new_name2 = str(local) +str(netname)+ "_arc.png"

# 绘图
plt.plot(x, y1, label="train average Loss")
plt.plot(x, y3, label="test average Loss")
plt.title("loss")
plt.xlabel('step')
plt.ylabel('probability')
plt.legend()  # 显示标签
plt.savefig(new_name1)
plt.show()


# 绘图
plt.plot(x, y2, label="train average accuracy")
plt.plot(x, y4, label="test average accuracy")
plt.title("accuracy")
plt.xlabel('step')
plt.ylabel('probability')
plt.legend()  # 显示标签
plt.savefig(new_name2)
plt.show()

classes=['Awake','Fatigue']
# 使用sklearn工具中confusion_matrix方法计算混淆矩阵
confusion_mat = confusion_matrix(label_train, label_train_pred)
print("confusion_mat.shape : {}".format(confusion_mat.shape))
print("confusion_mat : {}".format(confusion_mat))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
disp.plot(
    include_values=True,  # 混淆矩阵每个单元格上显示具体数值
    cmap="viridis",  # 不清楚啥意思，没研究，使用的sklearn中的默认值
    ax=None,  # 同上
    xticks_rotation="horizontal",  # 同上
    values_format="d"  # 显示的数值格式
)
plt.savefig("confusion_mat.png")
plt.show()
