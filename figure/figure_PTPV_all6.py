#!/usr/bin/env python
# coding: utf-8

from netCDF4 import Dataset
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import numpy as np
import cv2
import seaborn as sns  # 修改颜色条

deg = u'\xb0'
CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

shape1 = ['gradsameleadmean','samelead','bm1gradmean.gdat','lead1']     #alltransgrad   gradsameleadmean
shape2 = ['alltransgrad','singlemodel','bm7mon1gradmean.gdat','lead7mon1']             #singlemodel    samelead
tmp = shape2
#name = tmp[3]



#for l in range(10):
l = 7

target = str(1)

result = 'bm'+ str(l) + 'mon'+ target + 'gradmean.gdat'

ipth1 = 'E:/hyyc/pytorchCNN/PTPV/'+tmp[0]+'/'       #数据文件夹
ipth2 = 'E:/hyyc/pytorchCNN/PTPV/'+tmp[1]+'/all6/'       #保存结果文件夹

f = open(ipth1 + result, 'r')      #bm1mon12gradmean.gdat  bm1gradmean.gdat

heat_each = np.fromfile(f, dtype=np.float32).reshape(6, 24, 72)  # (12,36)  (6,18)

tmp = heat_each[3:]         #取sst或hc

for j in range(3):     #分开画多通道热图，时间维度分析
    name = 'lead' + str(l) + 'monhc'+str(j+1)

    #heat_each1 = np.mean(heat_each, axis=0)
    heat_each1 = tmp[j]

    print(heat_each1.max())
    heat_each1 = heat_each1 - np.min(heat_each1)
    heat_each1 = heat_each1 / (np.max(heat_each1))  # 化到0-1之间   分母可以加上1e-7避免分母为0，不过此问题肯定不会为0


    # heatmap: 36x6x18 -> 36x6x19,          axis表示沿着/在哪一维添加

    # ext_heatmap = np.append(heat_each1,heat_each1[:,0:1],axis=1) #4x4
    #ext_heatmap = np.append(heat_each1, heat_each1[:, 0:2], axis=1)  # 2x2
    ext_heatmap = np.append(heat_each1,heat_each1[:,0:4],axis=1) #1x1

    # standard deviation 标准差 (36x6x19 -> 6x19)
    std_heatmap = np.std(ext_heatmap, axis=0)
    # 求第0维的标准差，可以看作是求6*19各自对应点36个数的标准差，会得到6*19大小的数组

    # ext_heatmap = abs(ext_heatmap)
    # mean heatmap (36x6x19 -> 6x19) 36年平均
    mean_heatmap = np.mean(ext_heatmap, axis=0)

    '''# significant test 显著性检验
    mask_heatmap = np.zeros((36,6,19),dtype=np.float32)
    for i in range(36):
      for j in range(6):
        for k in range(19): #abs返回绝对值  sqrt开平方
          #level计算检测统计量公式
          level = abs(ext_heatmap[i,j,k]-mean_heatmap[j,k])/(std_heatmap[j,k]/np.sqrt(36))
          if level > 2.56: # org:2.56; 1.69; 2.03 不同不同
            #反正大概就是当Z>Zα时，为真，Zα为什么取2.56与设定的α有关，此处作者没提
            mask_heatmap[i,j,k] = ext_heatmap[i,j,k]
    '''
    '''
    heatmap = ext_heatmap[15,:,:]
    #heatmap0 = cv2.resize(heatmap,(6,19))
    print(heatmap)
    #uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255。
    #要想将当前的数组作为图像类型来进行各种操作，就要转换到uint8类型
    #方法1 np.uint8(想要转化的变量)
    #这种可能会导致原数据大于255的数被截断，当然此问题不会出现大于255，此处值太小所以乘255
    heatmap = np.uint8(2000*heatmap)
    heatmap = cv2.resize(heatmap,dsize=(640,480),interpolation=cv2.INTER_NEAREST)
    
    #方法2 用cv2.normalize函数配合cv2.NORM_MINMAX，可以设置目标数组的最大值和最小值，
    #然后让原数组等比例的放大或缩小到目标数组，如下面的例子中是将所有数字等比例的放大或缩小到0–255范围的数组中，
    #在不确定数值最大值的时候推荐下面的方法
    
    
    cv2.normalize(heatmap, heatmap, 0, 255, cv2.NORM_MINMAX)
    
    
    heatmap = np.array([heatmap],dtype='uint8')
    heatmap = heatmap.copy()
    print(heatmap)
    
    
    map = cv2.imread('E:\hyyc\enso\Figure_1.png')
    print(map.shape)
    
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    img_heatmap= cv2.addWeighted(map,0.6,heatmap,0.4,0)
    cv2.imshow('title',img_heatmap)#必须要有窗口的名字 title
    cv2.waitKey(0)
    '''

    # ext_heatmap = np.maximum(ext_heatmap,0) #去掉小于0

    # temp = cv2.resize(ext_heatmap[15,:,:],dsize=(24,76),interpolation=cv2.INTER_LINEAR)

    # extent指定热图x和y轴的坐标范围，zorder表示画图先后，数字小的先画
    # clim（min，max）设置当前图像的颜色限制

    a, b = ext_heatmap.max(), ext_heatmap.min()
    a = 1 if a > 0.999 else a
    print(a, b)
    cax = plt.imshow(ext_heatmap, cmap='RdBu_r', clim=[b, a],
                     interpolation="bicubic", extent=[0, 380, 60, -55], zorder=1)
    # 只通过上边这个把坐标范围限定了之后热图就得到了，后面的cax，subplot之类的只是在调整整个子图的位置
    '''mean_heatmap = np.maximum(mean_heatmap,0)
    cax = plt.imshow(mean_heatmap, cmap='RdBu_r',clim=[-8,8],interpolation="bicubic", extent=[0,380,60,-55],zorder=1)
    print(mean_heatmap.max())
    '''

    # 也可加入参数 interpolation="bicubic" 或其他合适插值方法
    # origin='lower/upper'将数组的[0,0]索引放在轴的左上角或左下角。
    plt.gca().invert_yaxis()

    # llcrnrlat=左下角纬度,urcrnrlat右上角纬度；llcrnrlon左下角经度, urcrnrlon右上角经度
    map = Basemap(projection='cyl', llcrnrlat=-55, urcrnrlat=59, resolution='c',
                  llcrnrlon=20, urcrnrlon=380)
    map.drawcoastlines(linewidth=0.2)
    map.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=6.5,
                      color='grey', linewidth=0.2)  # 画纬线
    map.drawmeridians(np.arange(0., 380., 60.), labels=[0, 0, 0, 1], fontsize=6.5,
                      color='grey', linewidth=0.2)  # 画经线
    map.fillcontinents(color='silver', zorder=2)

    space = '                                                                     '
    plt.title('Lead '+str(l)+' month '+ target +' HC '+str(j+1)+ space + '[El Niño Heatmap]', fontsize=8, y=0.962, x=0.5)

    # plt.show()
    cax1 = plt.axes([0.08, 0.28, 0.72, 0.013])  # 是为了画颜色条的
    # cax = plt.axes([0.08, 0.28, 0.72, 0.013])#[左，下，宽，高]规定的矩形区域 定义子图 https://www.zhihu.com/question/51745620
    # 前两个参数，左，下表示轴域原点坐标
    # 在已有的 axes 上绘制一个Colorbar，颜色条。
    cbar = plt.colorbar(cax=cax1, orientation='horizontal')
    # 对颜色条上参数的设置
    cbar.ax.tick_params(labelsize=6.5, direction='out', length=2, width=0.4, color='black')

    # plt.tight_layout(h_pad=0,w_pad=-0.6)#调整子图减少堆叠
    plt.subplots_adjust(bottom=0.10, top=0.9, left=0.08, right=0.8)
    plt.savefig(ipth2  + name +'.png', dpi=500,bbox_inches='tight')#默认dpi=100 dpi=155时突然变方块
    #bbox_inches='tight' 去除周围空白
    print(str(l),str(j+1))
    plt.close()


