#coding=gbk
'''
Created on 2013-5-3

@author: songjm
'''
import numpy as np
import scipy.signal as signal
import cPickle
import gzip
import gc
#import objgraph  
def sigmoid(input):
    '激励函数'
    out=1.7159*np.tanh(2.0/3.0*input)
    return out
def dsigmoid(input):
    'sigmoid的导函数已知input=sigmod(out)'
    out=2.0/3.0/1.7159*(1.7159+input)*(1.7159-input)
    return out
def convolutional(fm,ct,kernel,bias):
    '卷积函数'
    "fm 特征图 三维"
    "ct 连接权矩阵"
    "kernel 卷积核 三维"
    "bias 偏置 向量"
    map_width=fm.shape[2]
    map_height=fm.shape[1]
    kernel_width=kernel.shape[2]
    kernel_height=kernel.shape[1]
    '计算特征图的尺寸'
    dst_height=map_height-kernel_height+1
    dst_width=map_width-kernel_width+1
    '计算输出特征图的数量'
    cfmDims=np.max(ct[1])+1
    n_kernels=kernel.shape[0]
    #print("ct's shape%d:%d"%(ct.shape[0],ct.shape[1]))
    ' 初始化结果特征图' 
    cfm=np.zeros((cfmDims,dst_height,dst_width))
    for index in xrange(0,cfmDims):
        "首先加上偏置值"
        cfm[index]+=bias[index]
    for index in xrange(0,n_kernels):
        #print(index)
        this_fm=fm[ct[0,index]]
        this_kernel=kernel[index]
        "计算卷积"
        this_conv=signal.convolve2d(this_fm, this_kernel, mode='valid')
        cfm_index=ct[1,index]
        cfm[cfm_index]+=this_conv
    "使用sigmoid压制"
    return sigmoid(cfm)
def subsampling(fm,sW,sb,pool_size,pool_stride):
    "重采样函数"
    "fm 特征图"
    "sW 重采样权值 向量"
    "sb 重采样偏置 向量"
    "pool_size 重采样窗口大小 二维矩阵"
    "pool_stride 步长 int"
    sfm_width=int((fm.shape[2]-pool_size[1])/pool_stride)+1
    sfm_height=int((fm.shape[1]-pool_size[0])/pool_stride)+1
    sfmDims=fm.shape[0]
    sfm=np.zeros((sfmDims,sfm_width,sfm_height))
    #sfm=[]
    "使用权值为1的核进行采样"
    kernel=np.ones(pool_size)
    for index in xrange(0,sfmDims):
        this_fm=fm[index]
        this_kernel=kernel*sW[index]
        "采样实际就是一次卷积过程"
        this_sfm=signal.convolve2d(this_fm, this_kernel, mode='valid')
        sfm[index]=copy_fm(this_sfm,pool_stride)
        sfm[index]=sfm[index]+sb[index]
    sfm=np.array(sfm)
    sfm=sigmoid(sfm)
    return sfm
def copy_fm(fm,stride):
    height=fm.shape[0]
    width=fm.shape[1]
    result=[]
    for y in xrange(0,height,stride):
        y_result=[]
        y_data=fm[y]
        for x in xrange(0,width,stride):
            y_result.append(y_data[x])
        result.append(y_result)
    return np.array(result)
def max_with_index(value):
    "返回最大值，及最大值的下标"
    "结果 [下标,最大值]"
    d=np.max(value)
    i=np.argmax(value)
    return [i,d]
def grade(dout,out,input,w):
    dout=dout*dsigmoid(out)
    db=dout
    dw=np.zeros(w.shape)
    out=out*dsigmoid(out)
    for i in xrange(w.shape[1]):
        dw[:,i]=dout[i]*input
    din=np.zeros(input.shape)
    for i in xrange(db.shape[0]):
        this_kernel=w[:,i]
        #print("this kernrl:%s\n"%(this_kernel))
        this_dout=dout[i]*this_kernel
        #print("this dout: %s"%(this_dout))
        din+=this_dout
    return [din,dw,db]
def dconv2_in(dout,input,kernel):
    return signal.convolve2d(dout,kernel,mode='full')
def dconv2_kernel(dout,input,kernel):
    return signal.convolve2d(input,dout,mode='valid')
def initWeight(ct,kernel_shape):
    kernel_num=ct[0].shape[0]
    kernel_height=kernel_shape[0]
    kernel_width=kernel_shape[1]
    weights=np.zeros((kernel_num,kernel_height,kernel_width))
    for i in xrange(kernel_num):
        connected=np.array((ct[1]==ct[0,i]),dtype='int')
        connected=np.array(np.nonzero(connected))
        connected=connected.shape[0]
        fanin=connected*kernel_height*kernel_width
        sd=1.0/np.sqrt(fanin)
        weights[i]=-1.0*sd+2*sd*np.random.random_sample((kernel_height,kernel_width))
    return weights    
class convlayer:
    def __init__(self,cw,cb,sw,sb,ct,stride):
        self.sw=sw
        self.sb=sb
        self.cw=cw
        self.cb=cb
        self.ct=ct
        self.stride=stride
class bplayer:
    def __init__(self,n_in,n_out):
        sd=1.0/np.sqrt(n_in)
        self.w=-sd+2*sd*np.random.random_sample((n_in,n_out))
        self.b=np.zeros(n_out)
        self.n_in=n_in
        self.n_out=n_out
class convnet:
    def __init__(self,image_shape):
        self.eta=0.01
        self.decay=0.8
        self.step=2
        ct1=np.array([[1,1,1,1]])
        ct2=np.array([[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
                      [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                      [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                      [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]])
        ct1=np.transpose(ct1)
        ct2=np.transpose(ct2)
        conv_ct1=np.nonzero(ct1)
        conv_ct1=np.array((conv_ct1[1],conv_ct1[0]))
        conv_ct2=np.nonzero(ct2)
        conv_ct2=np.array((conv_ct2[1],conv_ct2[0]))
        image_height=image_shape[0]
        image_width=image_shape[1]
        self.image_shape=image_shape
        self.kernel_shape=[5,5]
        self.pool_shape=[2,2]
        self.stride=2
        stride=self.stride
        kernel_height=self.kernel_shape[0]
        kernel_width=self.kernel_shape[1]
        pool_height=self.pool_shape[0]
        pool_width=self.pool_shape[1]
        nofms1=ct1.shape[0]
        "初始化卷积层1"
        cmfHeight1=image_height-kernel_height+1
        cfmWidth1=image_width-kernel_width+1
        conv_layer1_cw=initWeight(conv_ct1, self.kernel_shape)
        conv_layer2_cb=np.zeros(nofms1)
        sfmHeight1=((cmfHeight1-pool_height)/stride)+1
        sfmWidth1=((cfmWidth1-pool_width)/stride)+1
        fanin=pool_height*pool_width
        sd=1.0/np.sqrt(fanin)
        conv_layer1_sw=-sd+2*sd*np.random.random_sample(nofms1)
        conv_layer1_sb=np.zeros(nofms1)
        self.conv1=convlayer(conv_layer1_cw,conv_layer2_cb,conv_layer1_sw,conv_layer1_sb,conv_ct1,stride)
        "初始化卷基层2"
        nofms2=ct2.shape[0]
        conv_layer2_cw=initWeight(conv_ct2, self.kernel_shape)
        conv_layer2_cb=np.zeros(nofms2)
        cfmHeight2=sfmHeight1-kernel_height+1
        cfmWidth2=sfmWidth1-kernel_width+1
        sfmHeight2=((cfmHeight2-pool_height)/stride)+1
        sfmWidth2=((cfmWidth2-pool_width)/stride)+1
        fanin=pool_width*pool_height
        sd=1.0/np.sqrt(fanin)
        conv_layer2_sw=-sd+2*sd*np.random.random_sample(nofms2)
        conv_layer2_sb=np.zeros(nofms2)
        self.conv2=convlayer(conv_layer2_cw,conv_layer2_cb,conv_layer2_sw,conv_layer2_sb,conv_ct2,stride)
        "BP分类器第一层"
        self.bp1=bplayer(nofms2*sfmHeight2*sfmWidth2,20)
        self.bp2=bplayer(20,10)
        self.noErrors=0
    def forwardPropConv(self,input):
        "开始第一层的卷积操作"
        ct=self.conv1.ct
        cw=self.conv1.cw
        cb=self.conv1.cb
        cfm1=convolutional(input, ct, cw, cb)
        sw=self.conv1.sw
        sb=self.conv1.sb
        sfm1=subsampling(cfm1, sw, sb, self.pool_shape, self.stride)
        "开始第二层的卷积操作"
        ct=self.conv2.ct
        cw=self.conv2.cw
        cb=self.conv2.cb
        cfm2=convolutional(sfm1, ct, cw, cb)
        sw=self.conv2.sw
        sb=self.conv2.sb
        sfm2=subsampling(cfm2, sw, sb, self.pool_shape, self.stride)
        return [cfm1,sfm1,cfm2,sfm2]
    def forwradPropBp(self,input):
        "开始第一层分类器的操作"
        w=self.bp1.w
        b=self.bp2.b
        out1=np.dot(input,w)
        for i in xrange(b.shape[0]):
            out1[i]+=b[i]
        out1=sigmoid(out1)
        "开始第二层的操作"
        w=self.bp2.w
        b=self.bp2.b
        out2=np.dot(out1,w)
        for i in xrange(b.shape[0]):
            out2[i]+=b[i]
        out2=sigmoid(out2)
        return [out1,out2]
    def backPropBp(self,sfm,tartget,fm1,fm2,w1,w2,b1,b2):
        "对BP层进行反向传播"
        "sfm 卷积层的S神经元的输出"
        "target 目标值"
        "fm1 BP层第一层输出"
        "fm2 BP层第二层输出"
        "w1 b1 第一层参数"
        "w2 b2 第二层参数"
        dtarget=-np.ones(self.bp2.n_out)*0.8
        dtarget[target]=0.8
        dfm2=np.zeros(self.bp2.n_out)
        dfm2=dfm2-dtarget
        dmf2=dfm2*dsigmoid(dfm2)
        dfm1,dw2,db2=grade(dfm2,fm2,fm1,w2)
        dsfm,dw1,db1=grade(dfm1,fm1,sfm,w1)
        return [dsfm,dw1,db1,dw2,db2]
    def backPropSubsampling(self,dsfm,sfm,cfm,sw,sb,pool_shape,stride):
        "对S神经元进行反向传播"
        "dsfm s神经元的输出的偏导数"
        "sfm s神经元的输出"
        "cfm 上层c神经元的输出"
        "sw sb 神经元参数"
        "pool_shape 采样窗口"
        "stride 采样步长"
        dims,height,width=cfm.shape
        dfm=np.zeros(cfm.shape)
        dsw=np.zeros(sw.shape)
        dsb=np.zeros(sb.shape)
        dsfm=dsfm*dsigmoid(sfm)
        kernel=np.ones(pool_shape)
        for i in xrange(dims):
            this_dsfm=dsfm[i]
            dthis_kernel=kernel*sw[i]
            dsb[i]=np.sum(this_dsfm)
            cfm_height=height-pool_shape[0]+1
            cfm_width=width-pool_shape[1]+1
            #print("height:%d,width:%d\n"%(cfm_height,cfm_width))
            dsfm_beforeSubsampling=np.zeros((cfm_height, cfm_width))
            y=0
            x=0
            for dy in xrange(0,dsfm_beforeSubsampling.shape[0],stride):
                x=0
                for dx in xrange(0,dsfm_beforeSubsampling.shape[1],stride):
                    dsfm_beforeSubsampling[dy,dx]=this_dsfm[y,x]
                    x+=1
                y+=1
            dfm[i]=dconv2_in(dsfm_beforeSubsampling,cfm[i],dthis_kernel)
            dsw[i]=np.sum(dthis_kernel)
        return [dfm,dsw,dsb]
    def backPropConvulution(self,dcfm,cfm,fm,ct,w,b):
        dfm=np.zeros(fm.shape)
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        dcfm=dcfm*dsigmoid(cfm)
        for i in xrange(b.shape[0]):
            this_dcfm=dcfm[i]
            db[i]=np.sum(this_dcfm)
        for i in xrange(w.shape[0]):
            this_fm=fm[ct[0,i]]
            this_w=w[i]
            this_dcfm=dcfm[ct[1,i]]
            this_dfm=dconv2_in(this_dcfm,this_fm,this_w)
            dfm[ct[0,i]]=dfm[ct[0,i]]+this_dfm
            dw[i]=dconv2_kernel(this_dcfm,this_fm,this_w)
        return [dfm,dw,db]
    def trian(self,input,target): 
        cfm1,sfm1,cfm2,sfm2=self.forwardPropConv(input)
        bp_input=sfm2.reshape(self.bp1.n_in)
        bp1_out,bp2_out=self.forwradPropBp(bp_input)
        out=max_with_index(bp2_out)
        if out[0]==target:
            self.noErrors+=1
        eta=self.eta
        w1=self.bp1.w
        b1=self.bp1.b
        w2=self.bp2.w
        b2=self.bp2.b
        dsfm,dw1,db1,dw2,db2=self.backPropBp(bp_input, target, bp1_out, bp2_out, w1, w2, b1, b2)
        self.bp1.w=self.bp1.w-dw1*eta
        self.bp1.b=self.bp1.b-db1*eta
        self.bp2.w=self.bp2.w-dw2*eta
        self.bp2.b=self.bp2.b-db2*eta
        dsfm=dsfm.reshape(sfm2.shape)
        sw2=self.conv2.sw
        sb2=self.conv2.sb
        ct2=self.conv2.ct
        cw2=self.conv2.cw
        cb2=self.conv2.cb
        pool_shape=self.pool_shape
        stride=self.stride
        dcfm2,dsw2,dsb2=self.backPropSubsampling(dsfm, sfm2,cfm2, sw2, sb2, pool_shape, stride)
        dsfm1,dcw2,dcb2=self.backPropConvulution(dcfm2, cfm2, sfm1, ct2, cw2, cb2)
        self.conv2.sw=self.conv2.sw-dsw2*eta
        self.conv2.sb=self.conv2.sb-dsb2*eta
        self.conv2.cw=self.conv2.cw-dcw2*eta
        self.conv2.cb=self.conv2.cb-dcb2*eta
        sw1=self.conv1.sw
        sb1=self.conv1.sb
        ct1=self.conv1.ct
        cw1=self.conv1.cw
        cb1=self.conv1.cb
        dcfm1,dsw1,dsb1=self.backPropSubsampling(dsfm1, sfm1,cfm1, sw1, sb1, pool_shape, stride)
        dfm,dcw1,dcb1=self.backPropConvulution(dcfm1, cfm1, input, ct1, cw1, cb1)
        self.conv1.sw=sw1-dsw1*eta
        self.conv1.sb=sb1-dsb1*eta
        self.conv1.cw=cw1-dcw1*eta
        self.conv1.cb=cb1-dcb1*eta
        #print(out)
def load_data(path):
    f=gzip.open(path,'rb')
    train_set,valid_set,test_set=cPickle.load(f)
    return [train_set,valid_set,test_set]
if __name__=='__main__':
    gc.enable()
    gc.set_debug(gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_INSTANCES | gc.DEBUG_OBJECTS)
    train_set,valid_set,test_set=load_data('d:/data/mnist.pkl.gz')
    num=train_set[0].shape[0]
    cnn=convnet([28,28])
    for train in xrange(1,21):
        cnn.noErrors=0
        print("第%d次训练"%(train))
        for i in xrange(1,num+1):
            image=train_set[0][i-1]
            data=image.reshape([1,28,28])
            target=train_set[1][i-1]
            cnn.trian(data, target)
            if i%1000==0:
                print("train %d,noErrors %d\n"%(i,cnn.noErrors)) 
                _unreachable = gc.collect()
                print("unreachable %d\n"%(_unreachable))
        if train%cnn.step==0:
            cnn.eta=cnn.eta*cnn.decay
            del data
            del image