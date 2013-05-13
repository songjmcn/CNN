#coding=gbk
'''
Created on 2013-5-11

@author: user
'''
import numpy
import scipy.signal as sg
from scipy import sum 
from scipy import dot
from scipy import tanh 
def sigmoid(data):
    '激活函数'
    out=1.7159*tanh(2.0/3.0*data)
    return out
def dsigmoid(data):
    out=2.0/3.0/1.7159*(1.7159+data)*(1.7159-data)
    return out
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
        '锟斤拷始锟斤拷锟斤拷'
        for dy in xrange(0,this_sfm.shape[0],ss):
            x=0
            for dx in xrange(0,this_sfm.shape[1],ss):
                sfm[i,y,x]=this_sfm[dy,dx]
                x+=1
            y+=1
        sfm[i]+=sb[i]
    sfm=sigmoid(sfm)
    return sfm
def conv3d(data,k):
        out=0
        for i in xrange(data.shape[0]):
            this_in=data[i]
            this_k=k[i]
            this_out=sg.convolve(this_in, numpy.rot90(this_k,2), 'valid')
            out+=this_out[0][0]
        return out
def classifer(sfm,k1,b1,k2,b2):
    'hidden layer'
    fm1Rows=sfm.shape[1]-k1.shape[2]+1
    fm1Cols=sfm.shape[2]-k1.shape[3]+1
    fm1Dims=b1.shape[0]
    fm1=numpy.zeros((fm1Dims))
    
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
    #print('dfm2 %s'%(dfm2))
    db2=numpy.zeros(b2.shape)
    dk2=numpy.zeros(k2.shape)
    dfm1=numpy.zeros(fm1.shape)
    'output layer'
    for i in xrange(db2.shape[0]):
        this_fm2=fm2[i]
        this_dfm2=dfm2[i]
        db2[i]=this_dfm2
        this_k2=k2[:,i]
        
        this_dfm1,dk=dOutputLayer(this_dfm2,this_fm2,fm1,this_k2)
        dk2[:,i]=dk
        dfm1=dfm1+this_dfm1
    'hidden layer'
    dfm1=dfm1*dsigmoid(fm1)
    db1=numpy.zeros(b1.shape)
    dsfm=numpy.zeros(sfm.shape)
    dk1=numpy.zeros(k1.shape)
    dfm1=dfm1.reshape((b1.shape[0],1,1))
    for i in xrange(b1.shape[0]):
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