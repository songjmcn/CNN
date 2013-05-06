#coding=gbk
'''
Created on 2013-5-6

@author: jianming song
'''
import theano
import theano.tensor as T
import numpy
import theano.tensor.nnet as conv
from theano.tensor.signal import downsample
import gzip
import cPickle
class LeyNetLayer:
    def __init__(self,image_shape,filter_shape,poolsize=(2,2)):
        '生成随机数生成器'
        rng = numpy.random.RandomState(23455)
        self.image_shape=image_shape
        self.filter_shape=filter_shape
        '每个特征图的输出'
        fan_in=numpy.prod(filter_shape[1:])
        self.poosize=poolsize
        '本层的输出'
        fan_out=(filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.w=theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),borrow=True)
        b_values=numpy.zeros((filter_shape[0]),dtype=theano.config.floatX)
        self.b=theano.shared(b_values,borrow=True)
        self.params=[self.w,self.b]
        return
    def evaluate(self,input):
        conv_out=conv.conv2d(input=input,filters=self.w,filter_shape=self.filter_shape,image_shape=self.image_shape)
        pooled_out=downsample.max_pool_2d(input=conv_out,ds=self.poosize,ignore_border=True)
        self.output=T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
class HiddenLayer:
    def __init__(self,n_in,n_out,active=T.nnet.sigmoid):
        rng=numpy.random.RandomState(23455)
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
        self.outout=self.active(tmp)
        return
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
        self.y_pred=T.argmax(self.p_y_given_x)
        return
    def negative_log_likelihood(self,y):
         t=T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
         return T.mean(t)
    def errors(self,input,target):
        self.evaluate(input)
        return T.mean(T.neq(self.y_pred, target))
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
def train_cnn(path='d:/data/mnist.pkl.gz.'):
    learn_rate=0.01
    batch_size=500
    datasets=load_data(path)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    layer1=LeyNetLayer(image_shape=[batch_size,1,28,28],filter_shape=[4,1,5,5])
    layer2=LeyNetLayer(image_shape=[batch_size,4,12,12],filter_shape=[16,1,5,5])
    layer3=HiddenLayer(n_in=16*4*4,n_out=20)
    layer4=OutLayer(n_in=20,n_out=10)
    def forward(x):
        layer1_in=x.reshape((batch_size,1,28,28))
        layer1.evaluate(x)
        layer2.evaluate(layer1.output)
        layer3_in=layer2.output.flatten(2)
        layer3.evaluate(layer3_in)
        layer4.evaluate(layer3.outout)
        return layer4.y_pred
    def update():
       cost=layer4.negative_log_likelihood(y)
       params = layer4.params + layer3.params + layer2.params + layer1.params
       grads=T.grad(cost,params)
       updates=[]
       for param_i, grad_i in zip(params, grads):
           updates.append((param_i,param_i-grad_i*learn_rate))
       return updates
    train_fun=theano.function(input=[index],output=forward,
                              givens={
                                      x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    for mini_batch in xrange(n_train_batches):
        target=train_fun(mini_batch)
        print(target)