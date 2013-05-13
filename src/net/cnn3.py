#coding=gbk
'''
Created on 2013-5-8

@author: songjm
'''
import numpy
import scipy.signal as sg
from scipy import dot
from scipy import sum
import gzip
import cPickle
from cnn_fun import *
'''
def sigmoid(data):
    '激活函数'
    out=1.7159*numpy.tanh(2.0/3.0*data)
    return out
def dsigmoid(data):
    out=2.0/3.0/1.7159*(1.7159+data)*(1.7159-data)
    return out
    '''
class ConvLayer:
    'convolution layer'
    def __init__(self):
        'init layer'
        self.cw=None
        self.cb=None
        self.ct=None
        self.sw=None
        self.sb=None
        self.skw=None
        self.ss=None
class Layer:
    'bp layer'
    def __init__(self):
        'init bp layer'
        self.w=None
        self.b=None
class LeyNet5:
    'convolution neuro network'
    def __init__(self):
        self.conv1=ConvLayer
        self.conv2=ConvLayer()
        self.classifer1=Layer()
        self.classifer2=Layer()
        self.eta=0.01
        self.decay=0.8
        self.no_errors=0.0
        self.total_conv=0.0
    def train(self,input,target):
        'train'
        pre_target=self.forwardProp(input)
        targetOut=-numpy.ones(self.classifer2.b.shape[0])*0.8
        targetOut[target]=0.8
        self.backProp(input, targetOut, self.eta)
        if pre_target==target:
            self.no_errors+=1
        self.total_conv+=1
    def clear(self):
        self.no_errors=0.0
        self.total_conv=0.0
        return
    def decay_eta(self):
        self.eta=self.eta*self.decay
    def test(self,input,target):
        ct=self.conv1.ct
        ck=self.conv1.cw
        cb=self.conv1.cb
        cfm1=convolutional(input, ct, ck, cb)
        sk=self.conv1.sw
        sb=self.conv1.sb
        skw=self.conv1.skw
        ss=self.conv1.ss
        sfm1=subsampling(cfm1, sk, sb, skw, ss)
        self.sfm1=sfm1
        ct=self.conv2.ct
        ck=self.conv2.cw
        cb=self.conv2.cb
        cfm2=convolutional(sfm1, ct, ck, cb)
        sk=self.conv2.sw
        sb=self.conv2.sb
        skw=self.conv2.skw
        ss=self.conv2.ss
        sfm2=subsampling(cfm2, sk, sb, skw, ss)
        k1=self.classifer1.w
        b1=self.classifer1.b
        k2=self.classifer2.w
        b2=self.classifer2.b
        fm1,fm2=classifer(sfm2, k1, b1, k2, b2)
        self.fm1=fm1
        self.fm2=fm2
        pre_target =numpy.argmax(fm2)
        self.total_conv+=1
        if pre_target==target:
            self.no_errors+=1
    def forwardProp(self,input):
        self.input=input
        ct=self.conv1.ct
        ck=self.conv1.cw
        cb=self.conv1.cb
        cfm1=convolutional(input, ct, ck, cb)
        self.cfm1=cfm1
        sk=self.conv1.sw
        sb=self.conv1.sb
        skw=self.conv1.skw
        ss=self.conv1.ss
        sfm1=subsampling(cfm1, sk, sb, skw, ss)
        self.sfm1=sfm1
        ct=self.conv2.ct
        ck=self.conv2.cw
        cb=self.conv2.cb
        cfm2=convolutional(sfm1, ct, ck, cb)
        self.cfm2=cfm2
        sk=self.conv2.sw
        sb=self.conv2.sb
        skw=self.conv2.skw
        ss=self.conv2.ss
        sfm2=subsampling(cfm2, sk, sb, skw, ss)
        self.sfm2=sfm2
        k1=self.classifer1.w
        b1=self.classifer1.b
        k2=self.classifer2.w
        b2=self.classifer2.b
        fm1,fm2=classifer(sfm2, k1, b1, k2, b2)
        self.fm1=fm1
        self.fm2=fm2
        return numpy.argmax(fm2)
    def backProp(self,fmaps,targetOut,eta):
        #dfm2=numpy.zeros(self.fm2.shape)
        dfm2=self.fm2-targetOut
        fm1=self.fm1
        fm2=self.fm2
        sfm=self.sfm2
        k1=self.classifer1.w
        b1=self.classifer1.b
        k2=self.classifer2.w
        b2=self.classifer2.b
        dsfm,dk1,db1,dk2,db2=bpClassifer(dfm2, fm1, fm2, sfm, k1, b1, k2, b2)
        self.classifer1.w=k1-dk1*eta
        self.classifer1.b=b1-db1*eta
        self.classifer2.w=k2-dk2*eta
        self.classifer2.b=b2-db2*eta
        sk=self.conv2.sw
        sb=self.conv2.sb
        skw=self.conv2.skw
        ss=self.conv2.ss
        sfm=self.sfm2
        cfm=self.cfm2
        dcfm,dsk,dsb=bpSubsampling(dsfm, sfm, cfm, sk, sb, skw, ss)
        self.conv2.sw=sk-dsk*eta
        self.conv2.sb=sb-dsb*eta
        sfm=self.sfm1
        ct=self.conv2.ct
        ck=self.conv2.cw
        cb=self.conv2.cb
        dsfm,dck,dcb=bpConvolutional(dcfm, cfm, sfm, ct, ck, cb)
        self.conv2.cw=ck-dck*eta
        self.conv2.cb=cb-dcb*eta
        cfm=self.cfm1
        sfm=self.sfm1
        sk=self.conv1.sw
        sb=self.conv1.sb
        skw=self.conv1.skw
        ss=self.conv1.ss
        dcfm,dsk,dsb=bpSubsampling(dsfm, sfm, cfm, sk, sb, skw, ss)
        self.conv1.sw=sk-dsk*eta
        self.conv1.sb=sb-dsb-eta
        fm=fmaps
        ck=self.conv1.cw
        cb=self.conv1.cb
        ct=self.conv1.ct
        dfm,dck,dcb=bpConvolutional(dcfm, cfm, fm, ct, ck, cb)
        self.conv1.cw=ck-dck*eta
        self.conv1.cb=cb-dcb*eta
'''
def dconv2d_in(dout,input,k,type):
    din=None
    if type=='valid':
        din=sg.convolve(dout,k,'full')
    elif type=='full':
        din=sg.convolve(dout,numpy.rot90(k,2),'valid')
    elif type=='same':
        din=sg.convolve(dout,k,'same')
    else:
        raise TypeError('type argument error')
    return din
def dconv2d_k(dout,input,k,type):
    dk=None
    if type=='valid':
        dk=sg.convolve(input, numpy.rot90(dout,2),'valid')
    elif type=='full':
        dk=sg.convolve(dout,numpy.rot90(dout,2),'valid')
    else:
        raise TypeError('type argument is error')
    return dk
'''
def create_cnn(image_shape=(28,28),filter_shape=(5,5),skw=2,stride=2):
    cnn=LeyNet5()
    ct1=numpy.transpose(numpy.array([[0,0,0,0],[0,1,2,3]]))
    ct2=numpy.transpose(numpy.array([[0,3,0,3,1,2,1,2,0,3,0,3,1,2,1,2,0,3,0,3,1,2,1,2,0,3,0,3,1,2,1,2],
                                     [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]]))
    fm_height=image_shape[0]
    fm_width=image_shape[1]
    filter_height=filter_shape[0]
    filter_width=filter_shape[1]
    def initWeight(ct,height,width):
        noDims=ct.shape[0]
        weights=numpy.zeros((noDims,height,width))
        for i in xrange(0,noDims):
            noFms=numpy.array(numpy.nonzero(numpy.array(ct[:,1]==ct[i,1],dtype='int')))
            noFms=noFms.shape[1]
            fanin=height*width*noFms
            sd=1./numpy.sqrt(fanin)
            weights[i]=-1*sd+2*sd*numpy.random.random_sample((height,width))
        return weights
    'C neuro'
    nofmsOne=numpy.max(ct1[:,1])+1
    cfmHeight1=fm_height-filter_height+1
    cfmWidth1=fm_width-filter_width+1
    cnn.conv1.cw=initWeight(ct1,filter_height,filter_width)
    cnn.conv1.cb=numpy.zeros((nofmsOne))
    cnn.conv1.ct=ct1
    'S neuro'
    skw1=skw
    ss1=stride
    sfmHeight1=((cfmHeight1-skw1)/ss1)+1
    sfmWidth1=((cfmWidth1-skw1)/ss1)+1
    cnn.conv1.skw=skw1
    cnn.conv1.ss=ss1
    fanin=sfmHeight1*sfmWidth1
    sd=1./numpy.sqrt(fanin)
    cnn.conv1.sw=-sd+2*sd*numpy.random.random_sample((nofmsOne))
    cnn.conv1.sb=numpy.zeros((nofmsOne))
    'C'
    nofmsTwo=numpy.max(ct2[:,1])+1
    cfmHeight2=sfmHeight1-filter_height+1
    cfmWidth2=sfmWidth1-filter_width+1
    cnn.conv2.cw=initWeight(ct2,filter_height,filter_width)
    cnn.conv2.cb=numpy.zeros((nofmsTwo))
    cnn.conv2.ct=ct2
    'S'
    skw2=skw
    ss2=stride
    sfmHeight2=((cfmHeight2-skw2)/ss2)+1
    sfmWidth2=((cfmWidth2-skw2)/ss2)+1
    fanin=sfmHeight2*sfmWidth2
    sd=1./numpy.sqrt(fanin)
    cnn.conv2.skw=skw
    cnn.conv2.ss=ss2
    cnn.conv2.sw=-sd+2*sd*numpy.random.random_sample((nofmsTwo))
    cnn.conv2.sb=numpy.zeros((nofmsTwo))
    'hidden one'
    nofm1=20
    fanin=sfmHeight2*sfmWidth2*nofm1
    sd=1./numpy.sqrt(fanin)
    cnn.classifer1.w=-sd+2*sd*numpy.random.random_sample((nofm1,nofmsTwo,sfmHeight2,sfmHeight2))
    cnn.classifer1.b=numpy.zeros((nofm1))
    'output layer'
    nofm2=10
    fanin=nofm1
    sd=1./numpy.sqrt(fanin)
    cnn.classifer2.w=-sd+2*sd*numpy.random.random_sample((nofm1,nofm2))
    cnn.classifer2.b=numpy.zeros((nofm2))
    return cnn
'''    
def convolutional(fm,ct,k,b):
    'fm feature map,3D'
    'ct coeenction table'
    'k kernel'
    'b bias' 
    'return cfm'
    cfmRows=fm.shape[1]-k.shape[1]+1
    cfmCols=fm.shape[2]-k.shape[2]+1
    cfmDims=numpy.max(ct[:,1])+1
    cfm=numpy.zeros((cfmDims,cfmRows,cfmCols))
    for i in xrange(cfmDims):
        cfm[i]=cfm[i]+b[i]
    for i in xrange(k.shape[0]):
        this_fm=fm[ct[i,0]]
        this_k=k[i]
        this_conv=sg.convolve(this_fm, numpy.rot90(this_k,2), mode='valid')
        cfm[ct[i,1]]=cfm[ct[i,1]]+this_conv
    cfm=sigmoid(cfm)
    return cfm
def subsampling(fm,sk,sb,sw,ss):
    sfmRows=((fm.shape[1]-sw)/ss)+1
    sfmCols=((fm.shape[2]-sw)/ss)+1
    sfmDims=fm.shape[0]
    sfm=numpy.zeros((sfmDims,sfmRows,sfmCols))
    kernel=numpy.ones((sw,sw))
    for i in xrange(sfmDims):
        this_fm=fm[i]
        this_kernel=kernel*sk[i]
        this_sfm=sg.convolve(this_fm,numpy.rot90(this_kernel,2),'valid')
        x=0
        y=0
        '开始采样'
        for dy in xrange(0,this_sfm.shape[0],ss):
            x=0
            for dx in xrange(0,this_sfm.shape[1],ss):
                sfm[i,y,x]=this_sfm[dy,dx]
                x+=1
            y+=1
        sfm[i]+=sb[i]
    sfm=sigmoid(sfm)
    return sfm
def classifer(sfm,k1,b1,k2,b2):
    'hidden layer'
    fm1Rows=sfm.shape[1]-k1.shape[2]+1
    fm1Cols=sfm.shape[2]-k1.shape[3]+1
    fm1Dims=b1.shape[0]
    fm1=numpy.zeros((fm1Dims))
    def conv3d(data,k):
        out=0
        for i in xrange(data.shape[0]):
            this_in=data[i]
            this_k=k[i]
            this_out=sg.convolve(this_in, numpy.rot90(this_k,2), 'valid')
            out+=this_out[0][0]
        return out
    for i in xrange(fm1Dims):
        fm1[i]+=b1[i]
        this_fm1=conv3d(sfm,k1[i])
        fm1[i]+=this_fm1
    fm1=sigmoid(fm1)
    fm2Dims=b2.shape[0]
    fm2=numpy.zeros(fm2Dims)
    tmp=dot(fm1,k2)
   
    for i in xrange(fm2Dims):
        fm2[i]+=tmp[i]+b2[i]
    fm2=sigmoid(fm2)
    return fm1,fm2
def dOutputLayer(dout,out,input,k):
        din=numpy.zeros(input.shape)
        dk=numpy.zeros(k.shape)
        for i in xrange(input.shape[0]):
            this_in=input[i]
            this_k=k[i]
            this_dk=dout*this_in
            dk[i]=this_dk
            this_din=dout*this_k
            din[i]=this_din
        return din,dk
def dconv3d(dout,out,input,k):
        din=numpy.zeros(input.shape)
        dk=numpy.zeros(k.shape)
        for i in xrange(input.shape[0]):
            this_in=input[i]
            this_k=k[i]
            this_din=dconv2d_in(dout,this_in,this_k,'valid')
            din[i]=this_din
            this_dk=dconv2d_k(dout,this_in,this_k,'valid')
            dk[i]=this_dk
        return din,dk
def bpClassifer(dfm2,fm1,fm2,sfm,k1,b1,k2,b2):
    dfm2=dfm2*dsigmoid(fm2)
    db2=numpy.zeros(b2.shape)
    dk2=numpy.zeros(k2.shape)
    dfm1=numpy.zeros(fm1.shape)
    'output layer'
    for i in xrange(db2.shape[0]):
        this_fm2=fm2[i]
        this_dfm2=dfm2[i]
        db2[i]=this_dfm2
        this_k2=k2[:,i]
        this_dfm1,dk2[:,i]=dOutputLayer(this_dfm2,this_fm2,fm1,this_k2)
        dfm1=dfm1+this_dfm1
    'hidden layer'
    dfm1=dfm1*dsigmoid(fm1)
    db1=numpy.zeros(b1.shape)
    dsfm=numpy.zeros(sfm.shape)
    dk1=numpy.zeros(k1.shape)
    dfm1=dfm1.reshape((b1.shape[0],1,1))
    for i in xrange(b2.shape[0]):
        this_dfm1=dfm1[i]
        this_fm1=fm1[i]
        db1[i]=sum(this_dfm1)
        this_dsfm,dk1[i]=dconv3d(this_dfm1,this_fm1,sfm,k1[i])
        dsfm=dsfm+this_dsfm
    return dsfm,dk1,db1,dk2,db2
def bpSubsampling(dsfm,sfm,fm,sk,sb,sw,ss):
    noDims,noRows,noCols=fm.shape
    dfm=numpy.zeros((noDims,noRows,noCols))
    dsk=numpy.zeros(sk.shape)
    dsb=numpy.zeros(sb.shape)
    dsfm=dsfm*dsigmoid(sfm)
    kernel=numpy.ones((sw,sw))
    for i in xrange(noDims):
        this_dsfm=dsfm[i]
        this_kernel=kernel*sk[i]
        dsb[i]=sum(this_dsfm)
        dsfm_beforeSubsampling=numpy.zeros((noRows-sw+1,noCols-sw+1))
        x=0
        y=0
        for dy in xrange(0,noRows-sw+1,ss):
            x=0
            for dx in xrange(0,noCols-sw+1,ss):
                dsfm_beforeSubsampling[dy,dx]=this_dsfm[y,x]
                x+=1
            y+=1
        dfm[i]=dconv2d_in(dsfm_beforeSubsampling,fm[i],this_kernel,'valid')
        dthis_kernel=dconv2d_k(dsfm_beforeSubsampling,fm[i],this_kernel,'valid')
        dsk[i]=sum(dthis_kernel)
    return dfm,dsk,dsb
def bpConvolutional(dcfm,cfm,fm,ct,k,b):
    dfm=numpy.zeros(fm.shape)
    dk=numpy.zeros(k.shape)
    db=numpy.zeros(b.shape)
    dcfm=dcfm*dsigmoid(cfm)
    for i in xrange(b.shape[0]):
        this_dcfm=dcfm[i]
        db[i]=sum(this_dcfm)
    for i in xrange(k.shape[0]):
        this_fm=fm[ct[i,0]]
        this_k=k[i]
        this_dcfm=dcfm[ct[i,1]]
        this_dfm=dconv2d_in(this_dcfm,this_fm,this_k,'valid')
        dfm[ct[i,0]]=dfm[ct[i,0]]+this_dfm
        dk[i]=dconv2d_k(this_dcfm,this_fm,this_k,'valid')
    return dfm,dk,db
'''
def load_data(path):
    f=gzip.open(path,'rb')
    train_set,valid_set,test_set=cPickle.load(f)
    return [train_set,valid_set,test_set]
def train():
    cnn=create_cnn(image_shape=(28,28), filter_shape=(5,5), skw=2, stride=2)
    train_set,valid_set,test_set=load_data('d:/data/mnist.pkl.gz')
    num=train_set[0].shape[0]
    total_loop=20
    train_sample=train_set[0]*2-1
    train_label=train_set[1]
    test_sample=test_set[0]*2-1
    test_label=test_set[1]
    test_num=test_label.shape[0]
    for loop in xrange(1,total_loop+1):
        print("train %d loop"%(loop))
        for i in xrange(num):
            #print(train_sample[i])
            cnn.train(train_sample[i].reshape(1,28,28),train_label[i])
            #cnn.train(numpy.array([train_sample[i].reshape(28,28)]), train_label[i])
            i=i+1
            if i %2000==0:
                print('total train %d,on errors %d'%(cnn.total_conv,cnn.no_errors))
        cnn.clear()
        for i in xrange(test_num):
            cnn.test(test_sample[i].reshape(1,28,28), test_label[i])
        print('T:%s\n'%(cnn.no_errors/cnn.total_conv))
        cnn.clear()
        if loop%2==0:
            cnn.decay_eta()
train()