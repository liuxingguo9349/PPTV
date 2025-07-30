#!/usr/bin/env python
from netCDF4 import Dataset
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.ticker as mtick
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import numpy as np

from brokenaxes import brokenaxes

deg = u'\xb0'

CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']


meancorpycnn = np.zeros((23))
meancorcnn = np.zeros((23))

ipth = 'E:/hyyc/pytorchCNN/result/allretrainresulttrans/'       #原pyCNN结果
ipth1 = 'E:/hyyc/pytorchCNN/retrain/allretrainresulttrans/'     #重训练结果
#ipth2 = 'E:/hyyc/pytorchCNN/retrain/'
ipth2 = 'E:/hyyc/pytorchCNN/'

#ipth2 = 'E:/hyyc/cnnresult/'  # 原cnn预测结果文件夹
# ipth3 = 'E:/hyyc/retrain/allretrainresult/'             #0.1-0.9重训练结果

# for l in range(23):
#     lead = l + 1

relationpycnn = np.zeros((12))
relationretrain = np.zeros((12))

lead = 1
for t in range(12):
    target = t+1

    lmont = str(lead)+'mon'+str(target)

    #retrain result
    retrain = open(ipth1 + 'fm' +lmont+'meanresult.gdat', 'r')
    retrain = np.fromfile(retrain,dtype=np.float32).reshape(36)

    pycnn = open(ipth + 'bm' +lmont+'meanresult.gdat', 'r') #原cnn模型全部数据训练预测结果
    pycnn = np.fromfile(pycnn,dtype=np.float32)

    # Open observation (GODAS, 1981-2017)
    f = Dataset('E:/hyyc/enso/input/input/GODAS.label.12mon.1982_2017.nc', 'r')
    obs = f.variables['pr'][:, t, 0, 0]
    '''36年 target月份'''

    pycnn = pycnn / np.std(pycnn)

    retrain = retrain / np.std(retrain)       #cnn一维数组,std标准差

    obs = obs / np.std(obs)
    # Compute correlation coefficient (1984-2017)  相关系数

    cor_pycnn = np.round(np.corrcoef(obs[3:], pycnn[3:])[0, 1], 2)

    cor_retrain = np.round(np.corrcoef(obs[3:], retrain[3:])[0, 1], 2)
    #round(x,2)返回浮点数x的四舍五入值,保留2位小数 corrcoef求相关系数
    #corrcoef得到的是方形矩阵，大小为两个被求矩阵行数的和，结果为两行数据之间的相关性
    #比如把两个2*2大小的矩阵划分为0-3共4个行，得到的相关性矩阵为2+2=4的4行4列的矩阵
    #如第一行为00,01,02,03行的相关性，第二行为10,11,12,13....
    #所以得到的相关性矩阵是对角线为1，的对称矩阵（因为0和1,1和0的相关性一致）


    relationpycnn[t] = cor_pycnn
    relationretrain[t] = cor_retrain

    '''meancorcnn[lead] = np.mean(relationcnn)
    meancormul[lead] = np.mean(relationmul)
    
    sumcorcnn = np.round(np.sum(meancorcnn),2) #np.round 返回四舍五入的 2位小数值
    sumcormul = np.round(np.sum(meancormul),2)'''

# bax = brokenaxes(ylims=((0, 0.3), (0.8, 1)),  # 设置y轴裂口范围
#                  hspace=0.25,  # y轴裂口宽度
#                  #wspace=0.2,   x轴裂口宽度
#                  despine=False,  # 是否y轴只显示一个裂口
#                  diag_color='r',  # 裂口斜线颜色
#                  )
n, m = 3, 1     # 设置子图 图形比例
gs = gridspec.GridSpec(2,1, height_ratios= [n , m], hspace = 0.1)   #设置两个子图比例为3:1，间距为0.1

x = np.arange(1, 13)
ax = plt.subplot(gs[0,0:])
ax1 = plt.subplot(gs[1,0:], sharex = ax)
#figure, (ax, ax1) = plt.subplots(2, 1, sharex=False)  # 绘制两个子图 来做分割

#plt.subplots_adjust(wspace=0,hspace=0.08) # 设置 子图间距
ax.plot(x, relationpycnn,'black',linewidth=1.5,marker='o', markersize=2,label='CNN+Scale')  #这里的label没用

ax.plot(x, relationretrain, 'orangered',linewidth=1.5, marker='o', markersize=2,label='Retrain')
ax1.plot([], [],'black',linewidth=1.5,marker='o', markersize=2,label='CNN+Scale')
ax1.plot([],[], 'orangered',linewidth=1.5, marker='o', markersize=2,label='Retrain')

ax.xaxis.set_visible(False)         #设置子图1的x坐标刻度不可见
ax.set_ylim(0.7,1)  # 设置纵坐标范围  第一个子图
ax1.set_ylim(0, 0.2)  # 设置纵坐标范围  第二个子图

ax1.xaxis.set_ticks(x)
#ax.yaxis.set_ticks([0.8,0.9,1])
#ax1.yaxis.set_ticks([0.00,0.10,0.20])
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))     #保留两位小数 为了和上面的子图一致

#ax.xaxis.set_major_locator(plt.NullLocator()) # 删除上边子图坐标轴的刻度显示

ax.spines['top'].set_visible(True)    # 边框控制
ax.spines['bottom'].set_visible(False) # 边框控制
ax.spines['right'].set_visible(True)  # 边框控制

ax1.spines['top'].set_visible(False)   # 边框控制
ax1.spines['bottom'].set_visible(True) # 边框控制
ax1.spines['right'].set_visible(True)  # 边框控制


plt.xlabel('Target months', fontsize=9)
ax.set_ylabel('Correlation skill', y=0.3, fontsize=9)

ax.tick_params(labeltop='off')

# 绘制断层线

d = 0.01  # 断层线的大小
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax1.transAxes, color='k')  # switch to the bottom axes
ax1.plot((-d, +d), (1 - d*3, 1 + d*3), **kwargs)  # bottom-left diagonal
ax1.plot((1 - d, 1 + d), (1 - d*3, 1 + d*3), **kwargs)  # bottom-right diagonal


ax.set_title('lead ' + str(lead) + ' month Correlation', fontsize=9)
ax.tick_params(labelsize=9)
ax1.tick_params(labelsize=9)


plt.legend(('CNN+LA+Scale','Retrain'), loc='lower left', prop={'size': 8}, ncol=3)

#plt.show()
plt.savefig(ipth2+'lead'+ str(lead) +'.png', dpi=500,bbox_inches='tight') #bbox_inches='tight' 去除周围空白)
plt.close()
print('lead ',lead)
'''
    #plt.subplot(2, 1, 1)
    x = np.arange(1, 13)
    #x = np.arange(2, 102)
    #y = np.arange(4, 38)
    lines = plt.plot(x, relationpycnn, 'black', x, relationretrain, 'orangered')#y, sin, 'dodgerblue'
    my_plot = plt.gca()
    line0 = my_plot.lines[0]
    line1 = my_plot.lines[1]
    #line2 = my_plot.lines[2]
    plt.setp(line0, linewidth=1.5,marker='o', markersize=2)
    plt.setp(line1, linewidth=1.5, marker='o', markersize=2)
    #plt.setp(line2, linewidth=0.5, marker='v', markersize=2)
    
    # plt.legend(('CNN (Sum=' + str(sumcor_cnn) +')',shape+' (Sum='+str(sumcor_bn)+')'),
    #            loc='lower left', prop={'size': 7}, ncol=3)
    
    plt.legend(('CNN+Scale','Retrain'), loc='lower left', prop={'size': 7}, ncol=3)   #loc='upper right'
    
    plt.xlim([0, 13])          #x轴作图范围
    plt.ylim([0, 1])           #这两行代码可删，不过画的线位置会有一点点不同，
    #x_tick = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    #plt.xticks(x,x_tick, fontsize=6.5)#1x,画图范围（间隔）2画图数据
    
    plt.xticks(x,x, fontsize=6.5)
    plt.yticks(np.arange(0, 1, 0.1), fontsize=6.5)
    
    # plt.xticks(x, x, fontsize=6.5)  #刻度内容
    # plt.yticks(np.arange(0, 1, 0.1), fontsize=6.5) #刻度内容
    
    plt.tick_params(labelsize=6., direction='in', length=2, width=0.3, color='black')#刻度线内容
    
    #plt.grid(linewidth=0.2, alpha=0.7)#显示网格线
    
    #plt.axhline(cor_cnn, color='black', linewidth=0.5)#绘制水平参考线，第一个参数代表画水平线y轴位置
    
    plt.title('lead '+str(lead)+' month Correlation', fontsize=8)
    
    plt.xlabel('Target months', fontsize=7)
    plt.ylabel('Correlation skill', fontsize=7)
    
    #显示每个点数据,plt.text(x坐标，y坐标，数据，其他参数）
    # for a in x:
    #     plt.text(a,relationmul[a-1],relationmul[a-1],horizontalalignment='center', color = 'red',fontsize = 6.5)
    # 
    plt.savefig(ipth2+'lead'+ str(lead) +'.png', dpi=500,bbox_inches='tight') #bbox_inches='tight' 去除周围空白)
    plt.close()
    print('lead ',lead)'''
