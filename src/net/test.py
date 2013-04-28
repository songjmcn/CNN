#coding=gbk
"author :宋建明"
"create data 2013-4-27"
import cPickle
import gzip
import layer
def load_data(path):
    f=gzip.open(path,'rb')
    train_set,valid_set,test_set=cPickle.load(f)
    return [train_set,valid_set,test_set]
def test_cnn():
    train_set,valid_set,test_set=load_data('d:/data/mnist.pkl.gz')
    layers=layer.CNN([28,28],[5,5])
    img=train_set[0][0]
    target=train_set[1][0]
    img=img.reshape([28,28])
    layers.forwardProp([img])
    print(target)
test_cnn()