'''
Created on 2013-5-7

@author: songjm
'''
import theano
import numpy
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal import downsample
import scipy
def test_3dcnn():
    theano.config.blas.ldflags="-lopenblas"
    a=numpy.random.random_sample((1,28,28,10,1))
    kernels=numpy.ones((3,5,5,3,1))
    skernels=numpy.ones((3,2,2,1,3))
    b_values=numpy.zeros((3))
    data=theano.shared(a,borrow=True)
    k=theano.shared(kernels,borrow=True)
    b=theano.shared(b_values,borrow=True)
    sk=theano.shared(skernels,borrow=True)
    sb=theano.shared(numpy.zeros((3)),borrow=True)
    d=theano.shared(numpy.array([1,1,1]),borrow=True)
    sd=theano.shared(numpy.array([2,2,1]),borrow=True)
    conv_out=T.nnet.conv3D(V=data,W=k,b=b,d=d)
    pool_out=T.nnet.conv3D(V=conv_out,W=sk,b=sb,d=sd)
    fun=theano.function(inputs=[],outputs=pool_out)
    out=fun()
    print(out.shape)
test_3dcnn()
class ConvLayer:
    def __init__(self,rng,fms_shape,filter_shape):
        fan_in=numpy.prod(filter_shape[1:])
        fan_out=(filter_shape[0] * numpy.prod(filter_shape[2:]) /4)
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.w=theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),borrow=True)
        self.b=theano.shared(numpy.zeros(filter_shape[0]),borrow=True)
        self.params=[self.w,self.b]
    def evaluate(self,input):
        self.output=nnet.conv3D(V=input,W=self.w,b=self.b,d=(1,1,1))
class SubsampleLayer:
    def __init__(self,rng,fms_shape,filter_shape):
        fan_in=numpy.prod(filter_shape[1:])
        fan_out=(filter_shape[0] * numpy.prod(filter_shape[2:]) /4)
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.w=theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),borrow=True)
        self.b=theano.shared(numpy.zeros(filter_shape[0]),borrow=True)
        self.params=[self.w,self.b]
    def evaluate(self,input):
        self.output=nnet.conv3D(V=input,W=self.w,b=self.b,d=(2,2,1))
class HiddenLayer:
    def __init__(self,rng,n_in,n_out,active=T.nnet.sigmoid):
        #rng=numpy.random.RandomState(23455)
        W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
        if active==T.nnet.sigmoid:
            W_values*=4
        self.w=theano.shared(value=W_values, name='W', borrow=True)
        b_values=numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b=theano.shared(value=b_values,borrow=True)
        self.active=active
        self.params=[self.w,self.b]
        return
    def evaluate(self,input):
        tmp=T.dot(input, self.w)+self.b
        self.output=self.active(tmp)
class OutLayer:
    def __init__(self,n_in,n_out):
        self.w = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.params=[self.w,self.b]
        return
    def evaluate(self,input):
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.w)+self.b)
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
        return
    def negative_log_likelihood(self,y):
         t=T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
         #theano.pp(self.errors(y))
         return -T.mean(t)
    def errors(self,target):
        return T.mean(T.neq(self.y_pred, target))
def load_data(path):
    