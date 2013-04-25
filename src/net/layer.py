#coding=gbk
'''
Created on 2013-4-25

@author: 宋健明
'''
import numpy as np
import scipy.signal as signal
import tools
class layer:
    def __init__(self,map_size,kernel_size,conn_t,pool_size=[2,2]):
        "this is init the layer"
        "map_size 输入的数据的尺寸 ,array list or tuple ,like[width,height]"
        "kernel_size 卷积核尺寸 ,array list or tuple ,like[width,height]"
        "conn_map 连接表 ,array list or tuple"
        "pool_size 二次采样核尺寸,array list or tuple,like[width,height]"
        
        map_size=np.array(map_size)
        kernel_size=np.array(kernel_size)
        conn_t=np.array(conn_t)
        width=map_size[0]
        height=map_size[1]
        k_width=kernel_size[0]
        k_height=kernel_size[1]
        "计算卷积后的特征图尺寸"
        fm_width=width-k_width+1    
        fm_height=height-k_height+1
        fm_num=conn_t.shape[1]
        self.map_size=map_size
        "计算连接表"
        self.ct=np.array(np.nonzero(conn_t))
        self.kernel_size=kernel_size
        #self.ct=np.transpose(self.ct)
        "初始化卷积核"
        self.cW=self.__initWidth__(self.ct, kernel_size)
        self.cb=np.zeros(fm_num)
        self.pool_size=pool_size
        "计算二次采样的核尺寸"
        sfm_height=int((fm_height-pool_size[1])/2)+1
        sfm_width=int((fm_width-pool_size[0])/2)+1
        self.pool_strid=2
        fanin=pool_size[0]*pool_size[1]
        sd=1/np.sqrt(fanin)
        "初始化二次采样核"
        self.sW=-sd+2*sd*np.random.random_sample(fm_num)
        self.sb=np.zeros(fm_num)
    def __initWidth__(self,conn_t,kernel_size):
        "conn_t"
        "kernel_size"
        kernel_num=conn_t.shape[1]
        weights=[]
        for index in xrange(0,kernel_num):
            connected=np.array((conn_t[1]==conn_t[0,index]),dtype='int')
            connected=np.array(np.nonzero(connected))
            connected=connected.shape[0]
            fain=kernel_size[0]*kernel_size[1]*connected
            sd=1/np.sqrt(fain)
            w=np.random.random_sample(kernel_size)
            w=-1*sd+2*sd*w
            weights.append(w)
        weights=np.array(weights)
        return weights
    def __forward_p__(self,input,ct,kernel,bias):
        if input.shape!=self.map_size:
            raise Exception("the input's shape is not equal the map size")
        dst_height=self.map_size[1]-self.kernel_size[1]+1
        dst_width=self.map_size[0]-self.kernel_size[0]+1
        cfmDims=np.max(ct[1])
        n_kernels=kernel.shape[0]
        cfm=np.zeros((cfmDims,dst_height,dst_width))
        for index in xrange(0,cfmDims):
            self.cfm[index]+=bias[index]
        for index in xrange(0,n_kernels):
            this_fm=input[ct[0,index]]
            this_kernel=bias[index]
            this_conv=signal.convolve2d(this_fm, this_kernel, mode='valid')
            cfm[index]+=this_conv
        return tools.sigmod(cfm)
def test():
    a=layer(map_size=np.array([28,28]),kernel_size=np.array([5,5]),conn_t=np.array([[1,1,1,1]]))
test()