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


deg = u'\xb0'
'''CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']
'''
'''for ii in range(23):
    for j in range(12):
        lead = ii+1
        target = j+1
'''

CH = 'C30H30'
EN = 'EN2'

lead = 1
target = 12
model = str(lead)+'mon'+str(target)

ipth1 = 'E:/hyyc/deepdream/'+model+'/doubleconv/'+CH+EN
ipth2 = 'E:/hyyc/deepdream/'+model+'/doubleconv/'+CH+EN+'/'


# Open Heatmap of each case (1981-2016)  36年

f = open(ipth2+'doubleconv'+model+CH+EN+'deepdreaminput.gdat','r')
#allgradcombinationdel0，是（23,12,72,24）大小的

#heat_each[:,:,:] = np.fromfile(f, dtype=np.float32).reshape(36,18,6)#[:37,:,:]
                                     #reshape之后进行切片操作再赋值给前边的i
input_each = np.fromfile(f, dtype=np.float32).reshape(72,24,6)

for i in range(6):
    #读取或保存文件的时候，不能+小数，因为可能会由于精度的问题产生比如0.1变成0.100001的情况
    #把heat_each的第三和第四维度交换，数组内的值也按照同样的方式交换，18*6变成了6*18
    #比如原来在(1,30,15,5)位置的值，转换到了(1,30,5,15)。与reshape不同的是
    #可以理解为reshape不改变数据的排列顺序，swapaxes改变数据本来的排列顺序

    input_each1 = np.swapaxes(input_each[:,:,i],0,1)  #(24,72)

    #ext_heatmap = np.append(heat_each,heat_each[:,:,0:1],axis=2)axis表示沿着/在哪一维添加

    ext_input = np.append(input_each1,input_each1[:,0:4],axis=1)
    '''画图的时候是把所有的数据都画出来的，所以这里ext_heatmap因为append成了(24,72+4)，
    所以画出来的热图其实是横向往左压缩了的，与地图实际位置不是完全一致，但对于分析影响不大'''


    # In[17]:      可以调用matplotlib中的imshow（）函数来绘制热图

    #x，y是尺寸相同的数组，两个数组中同样位置的数据值组成坐标，
    # 生成网格点坐标矩阵，在此处无用
    #x, y  = np.meshgrid(np.arange(0,380,2.5), np.arange(-91.25,91.25,2.5))
    # shade1996-1980 mask_heatmap[1996-1981, :, :]
    #修改分辨率

    #temp = cv2.resize(ext_heatmap[15,:,:],dsize=(24,76),interpolation=cv2.INTER_LINEAR)

    #extent指定热图x和y轴的坐标范围，zorder表示画图先后，数字小的先画
    #clim（min，max）设置当前图像的颜色限制
    #标签1873-1972年，此处要看1968年的，应该是在第95
    a = ext_input.min()
    b = ext_input.max()
    print(a,b)
    cax = plt.imshow(ext_input, cmap='RdBu_r',clim=[a,b],
         interpolation="bicubic", extent=[0,380,60,-55],zorder=3)

    #只通过上边这个把坐标范围限定了之后热图就得到了，
    #使用所有的数据进行画图，比如此时ext_heatmap是扩张为76列的，那数据左右就会排列的更紧密一些，
    #后面的cax，subplot之类的只是在调整整个子图的位置
    #也可加入参数 interpolation="bicubic" 或其他合适插值方法
    #origin='lower/upper'将数组的[0,0]索引放在轴的左上角或左下角。
    plt.gca().invert_yaxis()

    #llcrnrlat=左下角纬度,urcrnrlat右上角纬度；llcrnrlon左下角经度, urcrnrlon右上角经度
    map = Basemap(projection='cyl', llcrnrlat=-55,urcrnrlat=59, resolution='c',
                  llcrnrlon=20, urcrnrlon=380)
    map.drawcoastlines(linewidth=0.2)
    map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=6.5,
                      color='grey', linewidth=0.2)#画纬线
    map.drawmeridians(np.arange(0.,380.,60.),labels=[0,0,0,1],fontsize=6.5,
                      color='grey', linewidth=0.2)#画经线
    map.fillcontinents(color='silver', zorder=2)

    if i <=2:
        name = 'sst'
        ii = i+1
    else:
        name = 't300'
        ii = i-2

    space = '                      '
    plt.title('doubleconv deepdream '+model+CH+EN+' '+name+' month '+str(ii)+space+' [Bigger Niño3.4]',fontsize=8, y=0.965,x=0.5)

    #plt.show()
    cax1 = plt.axes([0.08, 0.28, 0.72, 0.013]) #是为了画颜色条的
    #cax = plt.axes([0.08, 0.28, 0.72, 0.013])#[左，下，宽，高]规定的矩形区域 定义子图 https://www.zhihu.com/question/51745620
    #前两个参数，左，下表示轴域原点坐标
    #在已有的 axes 上绘制一个Colorbar，颜色条。
    cbar = plt.colorbar(cax=cax1, orientation='horizontal')
    #对颜色条上参数的设置
    cbar.ax.tick_params(labelsize=6.5,direction='out',length=2,width=0.4,color='black')

    #plt.tight_layout(h_pad=0,w_pad=-0.6)#调整子图减少堆叠
    plt.subplots_adjust(bottom=0.10, top=0.9, left=0.08, right=0.8)
    #plt.subplots_adjust调整的是热图和下边颜色条两个子图
    plt.savefig(ipth1 +'/'+name+str(ii)+'deepdreaminput.jpg',dpi=200)#默认dpi=100 dpi=155时突然变方块
    plt.close() #每次保存完数据要close关闭否则只能保存一张图片

    #print(beishu)
    print(model,(i))
