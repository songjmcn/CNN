#coding=gbk
'''
Created on 2013-4-25

@author: 宋健明'''
import numpy as np
import scipy.signal as signal
import fun
class convLayer:
    def __init__(self,map_size,kernel_size,conn_t,pool_size=[2,2]):
        "this is init the layer"
        "map_size   特征图尺寸,array list or tuple ,like[width,height]"
        "kernel_size  卷积核尺寸,array list or tuple ,like[width,height]"
        "conn_map 连接权矩阵,array list or tuple"
        "pool_size 重采样核，array list or tuple,like[width,height]"
        
        map_size=np.array(map_size)
        kernel_size=np.array(kernel_size)
        conn_t=np.array(conn_t)
        width=map_size[0]
        height=map_size[1]
        k_width=kernel_size[0]
        k_height=kernel_size[1]
        fm_width=width-k_width+1    
        fm_height=height-k_height+1
        fm_num=conn_t.shape[1]
        self.map_size=map_size
        self.ct=np.array(np.nonzero(conn_t))
        self.kernel_size=kernel_size
        #self.ct=np.transpose(self.ct)
        self.cW=self.__initWidth__(self.ct, kernel_size)
        self.cb=np.zeros(fm_num)
        self.pool_size=pool_size
        sfm_height=int((fm_height-pool_size[1])/2)+1
        sfm_width=int((fm_width-pool_size[0])/2)+1
        self.pool_strid=2
        fanin=pool_size[0]*pool_size[1]
        sd=1.0/np.sqrt(fanin)
        self.sW=-sd+2*sd*np.random.random_sample(fm_num)
        self.sb=np.zeros(fm_num)
        self.out_shape=[sfm_height,sfm_width]
    def __initWidth__(self,conn_t,kernel_size):
        "初始化卷积核"
        "conn_t 连接权图"
        "kernel_size 核大小"
        kernel_num=conn_t.shape[1]
        self.fm_num=np.max(conn_t[1])+1
        weights=[]
        for index in xrange(0,kernel_num):
            connected=np.array((conn_t[1]==conn_t[0,index]),dtype='int')
            connected=np.array(np.nonzero(connected))
            connected=connected.shape[0]
            fain=kernel_size[0]*kernel_size[1]*connected
            sd=1.0/np.sqrt(fain)
            w=np.random.random_sample(kernel_size)
            w=-1.0*sd+2.0*sd*w
            weights.append(w)
        weights=np.array(weights)
        return weights
    def run(self,input):
        #if input.shape[0]!=self.map_size[0] or input.shape[1]!=self.map_size[1]:
        #    raise Exception("the input's shape is not equal the map size")
        input=np.array(input)
        cfm=fun.conv2d(input, self.ct, self.cW, self.cb)
        sfm=fun.subsampling(cfm, self.sW, self.sb, self.pool_size, self.pool_strid)
        self.sfm=sfm
        self.cfm=cfm
        self.output=sfm
        return
class BpLayer:
    def __init__(self,n_in,n_out):
        "n_in 输入数据的个数"
        "n_out 输出，即神经元的个数"
        self.n_in=n_in
        self.n_out=n_out;
        sd=1.0/np.sqrt(n_in)
        self.W=-sd+2*sd*np.random.random_sample((n_in,n_out))
        self.b=np.zeros(n_out)
        return
    def run(self,input):
        if input.shape[0]!=self.n_in:
            raise Exception("input size is not equal n_in")
        r=np.dot(input,self.W)
        for index in xrange(0,self.n_out):
            b=self.b[index]
            r[index]+=b
        r=fun.sigmoid(r)
        self.out_put=r
        return
class CNN:
    def __init__(self,image_shape,kernel_shape):
        self.image_shape=image_shape
        self.kernel_shape=kernel_shape
        self.__init_CNNLayer()
        return
    def __init_CNNLayer(self):
        self.__cnn_layer1__ct__=np.array([[1,1,1,1]])
        self.__cnn_layer2_ct__=np.array([
                                       [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
                                       [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                                       [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                                       [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
                                       ])
        self.__cnn_layer1__=convLayer(self.image_shape,self.kernel_shape,self.__cnn_layer1__ct__)
        self.__cnn_layer2__=convLayer(self.__cnn_layer1__.out_shape,self.kernel_shape,self.__cnn_layer2_ct__)
        n_cnn_output=self.__cnn_layer2__.fm_num*self.__cnn_layer2__.out_shape[0]*self.__cnn_layer2__.out_shape[1]
        self.__bp_hide__=BpLayer(n_cnn_output,20)
        self.__bp_out__=BpLayer(20,10)
        return
    def forwardProp(self,input):
        self.__cnn_layer1__.run(input)
        self.__cnn_layer2__.run(self.__cnn_layer1__.output)
        cnn_layer2_output=self.__cnn_layer2__.output
        cnn_out_size=self.__cnn_layer2__.fm_num*self.__cnn_layer2__.out_shape[0]*self.__cnn_layer2__.out_shape[1]
        cnn_output=self.__cnn_layer2__.output.reshape((cnn_out_size,))
        self.__bp_hide__.run(cnn_output)
        self.__bp_out__.run(self.__bp_hide__.out_put)
        out=self.__bp_out__.out_put
        print(out)
        self.output=fun.max_with_index(out)
        print(self.output)
        return
    def backProp(self):
        return