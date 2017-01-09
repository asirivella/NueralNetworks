import theano.tensor as T
from theano import function
from theano import shared
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import os
import time
from pandas import DataFrame, Series
import pandas as pd
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import random as rn

def load_data(filename, percentage):
    
    #for breast cancer
    #data_df = DataFrame.from_csv(filename, index_col=False, header=1)
    
    #for forest type data
    data_df = DataFrame.from_csv(filename, index_col=False)
    
    rows = data_df.shape[0]
    train_size = (int)((percentage/100.00)*rows)
    #val_size = (int)((10.00/100.00)*rows)
    #print "train size at load data : "
    #print train_size
    
    row_generator1 = rn.sample(range(data_df.shape[0]) , train_size)
    #print "train df at load data : "
    #print row_generator1.size()
    
    #bcd
    train_df = data_df.values[row_generator1];
    train_set = [train_df[:, 1:10], train_df[:, 10]]
    data_df.drop(row_generator1 , inplace=True)
    valid_set = [data_df.values[:, 1:10], data_df.values[:, 10]]
    test_set = data_df.values[:,1:10],data_df.values[:,10]
    
    #opt
    #train_df = data_df.values[row_generator1];
    #train_set = [train_df[:, 0:64], train_df[:, 64]]
    #data_df.drop(row_generator1 , inplace=True)
    #valid_set = [data_df.values[:, 0:64], data_df.values[:, 64]]
    #test_set = [data_df.values[:,0:64],data_df.values[:,64]]
    
    #ft
    #train_df = data_df.values[row_generator1];
    #train_set = [train_df[:, 1:28], train_df[:, 0]]
    #data_df.drop(row_generator1 , inplace=True)
    #valid_set = [data_df.values[:, 1:28], data_df.values[:, 0]]
    #test_set = [data_df.values[:,1:28],data_df.values[:,0]]
    
    #wer
    #train_df = data_df.values[row_generator1];
    #train_set = [train_df[:, 0:11], train_df[:, 11]]
    #data_df.drop(row_generator1 , inplace=True)
    #valid_set = [data_df.values[:, 0:11], data_df.values[:, 11]]
    #test_set = [data_df.values[:,0:11],data_df.values[:,11]]
    
    #bs
    #train_df = data_df.values[row_generator1];
    #train_set = [train_df[:, 1:5], train_df[:, 0]]
    #data_df.drop(row_generator1 , inplace=True)
    #valid_set = [data_df.values[:, 1:5], data_df.values[:, 0]]
    #test_set = [data_df.values[:,1:5],data_df.values[:,0]]
    
    #for breast cancer
    #train_set = [data_df.values[0: train_size, 1:10], data_df.values[0:train_size, 10]]
    #valid_set = [data_df.values[train_size:, 1:10], data_df.values[train_size:, 10]]
    #test_set = data_df.values[:,1:10],data_df.values[:,10]
    
    #for forest type
    #train_set = [data_df.values[0: train_size, 10:28], data_df.values[0:train_size, 0]]
    #valid_set = [data_df.values[train_size:val_size, 10:28], data_df.values[train_size:, 0]]
    #test_set = [data_df.values[:,1:28],data_df.values[:,0]]
    
    #for optical digits
    #train_set = [data_df.values[0: train_size, 0:64], data_df.values[0:train_size, 64]]
    #valid_set = [data_df.values[train_size:, 0:64], data_df.values[train_size:, 64]]
    #test_set = [data_df.values[:,0:64],data_df.values[:,64]]
    
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x,test_set_y)]
    return rval
	
	
class LogisticRegression(object):
    
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
            
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            
        self.params = [self.W, self.b]
            
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
    def predict(self):
        return self.y_pred
		
		
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
               
                
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        
        # parameters of the model
        self.params = [self.W, self.b]
		
		
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        self.hiddenLayer1 = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=(n_hidden*3),
                                       activation=T.tanh)
        
        self.hiddenLayer2 = HiddenLayer(rng=rng, input=self.hiddenLayer1.output, n_in=(n_hidden*3),
                                       n_out=n_hidden, activation=T.tanh)
        
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=n_out)
        
        self.L1 = abs(self.hiddenLayer1.W).sum() \
                + abs(self.hiddenLayer2.W).sum() \
                + abs(self.logRegressionLayer.W).sum()
            
        self.L2_sqr = (self.hiddenLayer1.W ** 2).sum() \
                    + (self.hiddenLayer2.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()
            
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.logRegressionLayer.params
        self.predict = self.logRegressionLayer.predict
		
def execute(filename, percentage, learning_rate, n_epochs, batch_size, hidden_size):
    #learning_rate=0.025
    L1_reg=0.00
    L2_reg=0.0001
    #n_epochs=50
    directory=filename
    #batch_size=10
    n_hidden = hidden_size

    datasets = load_data(directory, percentage)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print train_set_x.get_value().shape[0]
    print valid_set_x.get_value().shape[0]

    #print train_set_y.shape.eval()

    # compute number of minibatches for training, validation and testing
    n_train_batches = (train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = (valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = np.random.RandomState(1000)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=train_set_x.get_value().shape[1],
                 n_hidden=n_hidden, n_out=train_set_x.get_value().shape[1])

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
        + L1_reg * classifier.L1 \
        + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    validate_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                     x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                     y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                     x: train_set_x[index * batch_size:(index + 1) * batch_size],
                     y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                        # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0

    epoch = 0
    done_looping = False
    least_val_error = np.inf
    this_validation_loss = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                
                if(this_validation_loss < least_val_error) :
                    least_val_error = this_validation_loss
                    
                
                #print('epoch %i, validation error %f %%' %
                #         (epoch, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

            if patience <= iter:
                done_looping = True
                break

    print('epoch %i, least validation error %f %%' %
                          (100, least_val_error * 100.))
    #print('epoch %i, last validation error %f %%' %
    #                      (100, this_validation_loss * 100.))
    

for i in xrange(5) :
    percentage = (i + 1)*10
    for k in xrange(4) :
        hid = (k + 4)*100
        print 'percentage : ' 
        print percentage
        print 'hidden layer nodes :'
        print hid
        execute('od.csv', percentage, 0.025, 50, 25, hid)
		
