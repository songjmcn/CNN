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
        '学习率'
        self.__eta__=0.01
        '学习效率的衰减程度'
        self.__decay__=0.8
        '训练多少趟进行衰减'
        self.__step__=2      
        return
    @property
    def eta(self):
        return self.__eta__
    @eta.setter
    def eta(self,value):
        self.__eta__=value
        return
    @property
    def decay(self):
        return self.__decay__
    @decay.setter
    def decay(self,value):
        self.__decay__=value
        return
    @property
    def step(self):
        return self.__step__
    @step.setter
    def step(self,value):
        self.__step__=value
        return
    @property
    def cnn_layer1(self):
        return self.__cnn_layer1__
    @property
    def cnn_layer2(self):
        return self.__cnn_layer2__
    @property
    def bp_hide(self):
        return self.__bp_hide__
    @property
    def bp_out(self):
        return self.__bp_out__
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
        "前向传播"
        "input 输入值"
        
        "卷积层训练"
        self.__cnn_layer1__.run(input)
        self.__cnn_layer2__.run(self.__cnn_layer1__.output)
        cnn_layer2_output=self.__cnn_layer2__.output
        cnn_out_size=self.__cnn_layer2__.fm_num*self.__cnn_layer2__.out_shape[0]*self.__cnn_layer2__.out_shape[1]
        cnn_output=self.__cnn_layer2__.output.reshape((cnn_out_size,))
        self.__cnn_out_size=cnn_out_size
        self.__bp_hide__.run(cnn_output)
        self.__bp_out__.run(self.__bp_hide__.out_put)
        out=self.__bp_out__.out_put
        print(out)
        self.output=fun.max_with_index(out)
        print(self.output)
        return
    def backProp(self,input,target):
        targetOut=-np.ones(self.bp_out.out_put.shape)
        targetOut[target]=1*0.8
        self.__backProp__(input, targetOut)
    def __backProp__(self,input,target):
        dfm2=np.zeros(self.bp_out.n_out)
        for i in xrange(0,self.bp_out.out_put.shape[0]):
            dfm2[i]=self.bp_out.out_put[i]-target[i]
        fm1=self.cnn_layer1.output
        fm2=self.cnn_layer2.output
        sfm=self.cnn_layer2.sfm
        W1=self.bp_hide.W
        b1=self.bp_hide.b
        W2=self.bp_out.W
        b2=self.bp_out.b
        bph_in=self.__cnn_layer2__.output.reshape(self.__cnn_out_size)
        bph_out=self.__bp_hide__.out_put
        bpo_out=self.__bp_out__.out_put
        d_bpo_w,d_bpo_b,d_bpo_in=CNN.backProp_bpLayer(bph_out, dfm2, bpo_out, W2, b2)
        d_bph_w,d_bph_b,d_bph_in=CNN.backProp_bpLayer(bph_out,d_bpo_in,bph_in,W1,b1)
        self.bp_out.W=W2-d_bpo_w*self.eta
        self.bp_out.b=b2-d_bpo_b*self.eta
        self.bp_hide.W=W1-d_bph_w*self.eta
        self.bp_hide.b=b1-d_bph_b*self.eta
    @staticmethod  
    def backProp_bpLayer(input,dtarget,output,w,b):
        "对bp层进行反向传播，调整参数"
        "input 输入数据"
        "dtaerget 期望的输出的导数"
        "input 本层的输出"
        "w 本层的权值"
        "b 本层的偏置值"
        "return [dw,db,din] 权值的梯度，偏置值的梯度，输入数据的梯度"
        dtarget=dtarget*fun.dsigmoid(output)
        db=np.zeros(b.shape)
        dw=np.zeros(w.shape)
        din=np.zeros(input.shape)
        for i in xrange(0,db.shape[0]):
            this_output=output[i]
            this_dtarget=dtarget[i]
            db[i]=np.sum(this_dtarget)
            this_w=w[i]
            this_din,dw[i]=CNN.dMultiply(dtarget,output,input,this_w)
            din=din+this_dtarget
        return [dw,db,din]
    @staticmethod
    def backProp_conv(dcfm,cfm,fm,ct,w,b):
        '将卷积层反向传播'
        'dcfm 误差关于cfm的偏导数'
        'cfm 卷基层的输出'
        'fm 卷基层的输入'
        'ct fm和cfm之间的连接表'
        'w 卷积核'
        'b 偏置'
        dfm=np.zeros(fm.shape)
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        dcfm=dcfm*fun.dsigmoid(cfm)
        for i in xrange(0,b.shape[0]):
            this_dcfm=dcfm[i]
            db[i]=np.sum(this_dcfm)
        for i in xrange(0,w.shape[0]):
            this_fm=fm[ct[0,i]]
            this_w=w[i]
            this_dcfm=dcfm[ct[1,i]]
            '误差关于dfm的偏导数'
            this_dfm=fun.dconv2(this_dcfm, this_w)
            dfm[ct[0,i]]=dfm[ct[0,i]]+this_dfm
            '误差关于dw的偏导数'
            dw[i]=fun.dconv2_kernel(this_dcfm, this_fm)
        return [dfm,dw,db]
    @staticmethod
    def backPropSubsampling(dsfm,sfm,fm,sw,sb,pool_size,stride):
        dims,height,width=fm.shape
        dfm=np.zeros((dims,height,width))
        dsw=np.zeros(sw.shape)
        dsb=np.zeros(sb.shape)
        dsfm=dsfm*fun.dsigmoid(sfm)
        kernel=np.ones(pool_size)
        for i in xrange(0,dims):
            this_dsfm=dsfm[i]
            this_kernel=kernel*sw[i]
            dsb[i]=np.sum(this_dsfm[:])
            dsfm_beforeSubsampling=np.zeros((height-pool_size[0]+1,width-pool_size[1]+1))
            for y in xrange(0,dsfm_beforeSubsampling.shape[0],stride):
                for x in xrange(0,dsfm_beforeSubsampling.shape[1],stride):
                    dsfm_beforeSubsampling[y,x]=this_dsfm
            dfm[i]=fun.dconv2(dsfm_beforeSubsampling, fm[i])
            dthis_kernel=fun.dconv2_kernel(dsfm_beforeSubsampling, fm[i])
            dsw[i]=np.sum(dthis_kernel)
        return [dfm,dsw,dsb]
    def train(self,input,target):
        self.forwardProp(input)
        self.backProp(input,target)
    @staticmethod
    def dMultiply(dout,out,input,w):
        din=np.zeros(input.shape)
        dw=np.zeros(w.shape)
        for i in xrange(0,input.shape[0]):
            this_in=input[i]
            this_w=w[i]
            this_dw=dout*this_in
            dw[i]=np.sum(this_dw)
            this_din=dout*this_w
            din[i]=this_din
        return np.array([din,dw])