import tflearn
import tensorflow as tf
import wsj_loader

def print_shape(x):
    print(x.shape)
    print(x[0].shape)


wsj = wsj_loader.WSJ()
train, dev, test = wsj.train, wsj.dev, wsj.test

print_shape(train)
print_shape(test)
print_shape(dev)

