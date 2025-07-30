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

'''删除海域梯度
inp1 = Dataset('E:/hyyc/enso/input/input/SODA.input.36mon.1871_1970.nc','r')

inpv1 = np.zeros((100,6,24,72), dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][0:100,22:25,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][0:100,22:25,:,:]
tmp1 = np.zeros((24,72))
for mm in range(24):
    for nn in range(72):
        bb = np.sum(inpv1[14,:,mm,nn])
        if bb == 0:
            bb = 0
        else:
            bb = 1
        tmp1[mm,nn] = bb
'''


lead = 1
for target in range(1,13,1):
    lmont = str(lead)+'mon'+str(target)
    lei = 'trans'
    change = 'noLSaddFC'

    ipth1 = 'E:/hyyc/ConvNeXt/addCBAM/allresult/'

    ipth2 = 'E:/hyyc/ConvNeXt/addCBAM/'


    # Open Heatmap of each case (1981-2016)  36年

    f = open(ipth1 +'1mon1gradCHmean.gdat','r')
    #allgradcombinationdel0，是（23,12,72,24）大小的

    #convNeXt输入就是（batch，6,24,72）
    '''原CNN是（72,24,6）大小的'''
    gradxmean = np.fromfile(f, dtype=np.float32).reshape(72,24,6)
    #heat_each1 = np.mean(gradxmean,axis=0)
    #for l in range(num):
    heat_each = gradxmean
    heat_each1 = np.mean(heat_each,axis=2)
    heat_each1 = np.swapaxes(heat_each1, 0, 1)  # (72,24)->(24,72)
    #heat_each1 = np.flipud(heat_each1)
    '''因为训练数据由上到下是，由南纬55->北纬60，画图的时候应该进行上下翻转'''

    ext_heatmap = np.append(heat_each1,heat_each1[:,0:4],axis=1)
    print('ext_heatmap shape:',ext_heatmap.shape)
    '''画图的时候是把所有的数据都画出来的，所以这里ext_heatmap因为append成了(24,72+4)，
    画出来的热图不是横向往左压缩了的，而是根据横坐标20-360,0-20画的，与地图实际位置对应'''

    # standard deviation 标准差 (36x6x19 -> 6x19)
    std_heatmap = np.std(ext_heatmap,axis=0)
    #求第0维的标准差，可以看作是求6*19各自对应点36个数的标准差，会得到6*19大小的数组

    # mean heatmap (36x6x19 -> 6x19) 36年平均
    mean_heatmap = np.mean(ext_heatmap,axis=0)


    # In[17]:      可以调用matplotlib中的imshow（）函数来绘制热图

    #temp = cv2.resize(ext_heatmap[15,:,:],dsize=(24,76),interpolation=cv2.INTER_LINEAR)

    '''extent指定热图x和y轴的坐标范围,数据按数据本身的顺序画出来，只是extent给这些位置的数据设置一个坐标
    与origin参数关联，origin默认为‘upper’，就是默认坐标轴左下角是x轴的起点，就是x轴的0，y轴默认是逆序，
    即y轴最上面坐标是0，下面坐标x轴0坐标的交点是y轴最大值，如果设置origin为lower，则会将y轴进行颠倒
    即y轴坐标原点也在左下角，同时会将y轴对应数据进行翻转，原来在第一行的到了最后一行'''
    #zorder表示画图先后，数字小的先画
    #clim（min，max）设置当前图像的颜色限制
    #标签1873-1972年，此处要看1968年的，应该是在第95

    a = ext_heatmap.max()
    print(a,ext_heatmap.min())
    cax = plt.imshow(ext_heatmap, cmap='RdBu_r',clim=[-a,a],
         interpolation="bicubic", extent=[0,380,60,-55],zorder=3)

    #只通过上边这个把坐标范围限定了之后热图就得到了，
    #使用所有的数据进行画图，比如此时ext_heatmap是扩张为76列的，那数据左右就会排列的更紧密一些，
    #后面的cax，subplot之类的只是在调整整个子图的位置
    #也可加入参数 interpolation="bicubic" 或其他合适插值方法

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

    space = '                                                      '
    plt.title(lmont+' P.T.P.V'+space+' [El Niño Heatmap]',fontsize=8, y=0.965,x=0.5)

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

    plt.savefig(ipth2 +lmont+'PTPV'+'.jpg',dpi=200)#默认dpi=100 dpi=155时突然变方块
    plt.close() #每次保存完数据要close关闭否则只能保存一张图片

    #print(beishu)
    print(lmont)
