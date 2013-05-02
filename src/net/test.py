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
    imgs=train_set[0].shape[0]
    train_num=10
    for train in xrange(train_num):
        print("第%d趟训练\n"%(train+1))
        completed=0
        for index in xrange(imgs):
            img=train_set[0][index]
            target=train_set[1][index]
            img=img.reshape([1,28,28])
            layers.train(img, target)
            completed+=1
            if completed%1000==0 and completed!=0:
                print('total sample %s,no errors %s\n'%(completed,layers.noErrors))
            
test_cnn()