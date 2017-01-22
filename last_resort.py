"""
Instruction:

In this section, you are asked to train a NN with different hyperparameters.
To start with training, you need to fill in the incomplete code. There are 3
places that you need to complete:
a) Backward pass equations for an affine layer (linear transformation + bias).
b) Backward pass equations for ReLU activation function.
c) Weight update equations with momentum.

After correctly fill in the code, modify the hyperparameters in "main()".
You can then run this file with the command: "python nn.py" in your terminal.
The program will automatically check your gradient implementation before start.
The program will print out the training progress, and it will display the
training curve by the end. You can optionally save the model by uncommenting
the lines in "main()".
"""

from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot
import sys
import numpy as np
from pylab import savefig
import matplotlib.pyplot as plt
import random, os
from os.path import sep
from scipy.misc import imread
from scipy.misc import imresize
from random import shuffle
import cPickle

##########################################################################
################################ MACROS ##################################
PATH_SEPARATOR = sep
CROPPED_PATH = "cropped"
ONE = "AngieHarmon"
TWO = "DanielRadcliffe"
THREE = "GerardButler"
FOUR = "LorraineBracco"
FIVE = "MichaelVartan"
SIX = "PeriGilpin"
##########################################################################
########################## HYPERPARAMETERS ###############################
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
X_DIM = 1024
Y_DIM = 6
##########################################################################
##########################################################################



def get_all_images(path):
    all_pics = []
    for filename in os.listdir(path):
        img = imread(path + sep + filename)
        all_pics.append(img)
    return all_pics

def big_data(images1, images2, images3, images4, images5, images6):
    big_data = []
    for thing1 in images1:
        big_data.append(thing1)
    for thing2 in images2:
        big_data.append(thing2)
    for thing3 in images3:
        big_data.append(thing3)
    for thing4 in images4:
        big_data.append(thing4)
    for thing5 in images5:
        big_data.append(thing5)
    for thing6 in images6:
        big_data.append(thing6)

    
    
    return big_data

def compress(images):
    im = []
    for image in images:
        im.append(np.ndarray.flatten(image))
    return np.array(im)

def decompress(images):
    im = []
    for image in images:
        im.append(image.reshape(32, 32))
    return np.array(im)

def one_hots(num, num_categories):
    one_hot_encoding = []
    for i in range(num_categories):
        if num == i+1:
            one_hot_encoding.append(1)
        else:
            one_hot_encoding.append(0)
    return one_hot_encoding

def x_y(list_of_x, y):
    couple = []
    for thing in list_of_x:
        couple.append([thing, y])
    return couple

def uncouple(list):
    one = []
    two = []
    for thing in list:
        one.append(thing[0])
        two.append(thing[1])
    return one, two

def InitNN(num_inputs, num_hiddens, num_outputs):
    """Initializes NN parameters.

    Args:
        num_inputs:    Number of input units.
        num_hiddens:   List of two elements, hidden size for each layer.
        num_outputs:   Number of output units.

    Returns:
        model:         Randomly initialized network weights.
    """
    W1 = 0.1 * np.random.randn(num_inputs, num_hiddens[0])
    W2 = 0.1 * np.random.randn(num_hiddens[0], num_hiddens[1])
    W3 = 0.01 * np.random.randn(num_hiddens[1], num_outputs)
    b1 = np.zeros((num_hiddens[0]))
    b2 = np.zeros((num_hiddens[1]))
    b3 = np.zeros((num_outputs))
    v_W1 = 0
    v_W2 = 0
    v_W3 = 0
    v_b1 = 0
    v_b2 = 0
    v_b3 = 0
    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'v_W1': v_W1,
        'v_W2': v_W2,
        'v_W3': v_W3,
        'v_b1': v_b1,
        'v_b2': v_b2,
        'v_b3': v_b3
    }
    
    #print("BEE ", model['W1'].T[0].shape)
    #raise Expection("Stop!")
    
    return model

def Affine(x, w, b):
    """Computes the affine transformation.

    Args:
        x: Inputs
        w: Weights
        b: Bias

    Returns:
        y: Outputs
    """
    y = x.dot(w) + b
    return y

def AffineBackward(grad_y, x, w):
    """Computes gradients of affine transformation.

    Args:
        grad_y: gradient from last layer
        x: inputs
        w: weights

    Returns:
        grad_x: Gradients wrt. the inputs.
        grad_w: Gradients wrt. the weights.
        grad_b: Gradients wrt. the biases.
    """
    grad_x = np.dot(grad_y, w.T)
    grad_w = np.dot(x.T, grad_y)
    ones = np.ones(grad_y.shape[0])
    grad_b = np.dot(grad_y.T, ones)
    
    return grad_x, grad_w, grad_b

def ReLU(x):
    """Computes the ReLU activation function.

    Args:
        x: Inputs

    Returns:
        y: Activation
    """
    return np.maximum(x, 0.0)


def ReLUBackward(grad_y, x, y):
    """Computes gradients of the ReLU activation function.
    Args: 
        x: before ReLu
        y: After ReLU
        grad_y: previous gradient
    Returns:
        grad_x: Gradients wrt. the inputs.
    """
    grad_y[x <= 0] = 0
    return grad_y

def Softmax(x):
    """Computes the softmax activation function.

    Args:
        x: Inputs

    Returns:
        y: Activation
    """
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def NNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    h1 = Affine(x, model['W1'], model['b1'])
    h1r = ReLU(h1)
    h2 = Affine(h1r, model['W2'], model['b2'])
    h2r = ReLU(h2)
    y = Affine(h2r, model['W3'], model['b3'])
    var = {
        'x': x,
        'h1': h1,
        'h1r': h1r,
        'h2': h2,
        'h2r': h2r,
        'y': y
    }
    return var


def NNBackward(model, err, var):
    """Runs the backward pass.

    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh2r, dE_dW3, dE_db3 = AffineBackward(err, var['h2r'], model['W3'])
    dE_dh2 = ReLUBackward(dE_dh2r, var['h2'], var['h2r'])
    dE_dh1r, dE_dW2, dE_db2 = AffineBackward(dE_dh2, var['h1r'], model['W2'])
    dE_dh1 = ReLUBackward(dE_dh1r, var['h1'], var['h1r'])
    _, dE_dW1, dE_db1 = AffineBackward(dE_dh1, var['x'], model['W1'])
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    pass


def NNUpdate(model, eps, momentum):
    """Update NN weights.

    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    
    model['v_W1'] = momentum * model['v_W1'] + eps * model['dE_dW1']
    model['v_W2'] = momentum * model['v_W2'] + eps * model['dE_dW2']
    model['v_W3'] = momentum * model['v_W3'] + eps * model['dE_dW3']
    model['v_b1'] = momentum * model['v_b1'] + eps * model['dE_db1']
    model['v_b2'] = momentum * model['v_b2'] + eps * model['dE_db2']
    model['v_b3'] = momentum * model['v_b3'] + eps * model['dE_db3']
    
    model['W1'] = model['W1'] - model['v_W1']
    model['W2'] = model['W2'] - model['v_W2']
    model['W3'] = model['W3'] - model['v_W3']
    model['b1'] = model['b1'] - model['v_b1']
    model['b2'] = model['b2'] - model['v_b2']
    model['b3'] = model['b3'] - model['v_b3']
    
    """
    model['W1'] = model['W1'] - eps * model['dE_dW1']
    model['W2'] = model['W2'] - eps * model['dE_dW2']
    model['W3'] = model['W3'] - eps * model['dE_dW3']
    model['b1'] = model['b1'] - eps * model['dE_db1']
    model['b2'] = model['b2'] - eps * model['dE_db2']
    model['b3'] = model['b3'] - eps * model['dE_db3']
    """


def Train(model, forward, backward, update, eps, momentum, num_epochs,
          batch_size):
    """Trains a simple MLP.

    Args:
        model:           Dictionary of model weights.
        forward:         Forward prop function.
        backward:        Backward prop function.
        update:          Update weights function.
        eps:             Learning rate.
        momentum:        Momentum.
        num_epochs:      Number of epochs to run training for.
        batch_size:      Mini-batch size, -1 for full batch.

    Returns:
        stats:           Dictionary of training statistics.
            - train_ce:       Training cross entropy.
            - valid_ce:       Validation cross entropy.
            - train_acc:      Training accuracy.
            - valid_acc:      Validation accuracy.
    """
    

    #print x_y([[1,2], [3,4]], 7)
    #print one_hots(1, 6)
    #print one_hots(6, 6)

    ## GET IMAGES AND TARGETS FOR IMAGE 1
    images1 = get_all_images(CROPPED_PATH + sep + ONE)
    plt.imshow(images1[0])
    plt.show()
    images1 = compress(images1)
    shuffle(images1)
    y1 = np.array(one_hots(1, 6))
    image1_test  = images1[0:10]
    image1_val   = images1[10:20]
    image1_train = images1[20:]
    image1_test_target = [y1] * 10
    image1_val_target = [y1] * 10
    image1_train_target = [y1] * len(image1_train)
    
    ## GET IMAGES AND TARGETS FOR IMAGE 2
    images2 = get_all_images(CROPPED_PATH + sep + TWO)
    images2 = compress(images2)
    shuffle(images2)
    y2 = np.array(one_hots(2, 6))
    #images2 = x_y(images2, y2)
    image2_test  = images2[0:10]
    image2_val   = images2[10:20]
    image2_train = images2[20:]
    image2_test_target = [y2] * 10
    image2_val_target = [y2] * 10
    image2_train_target = [y2] * len(image2_train)
    
    ## GET IMAGES AND TARGETS FOR IMAGE 3
    images3 = get_all_images(CROPPED_PATH + sep + THREE)
    images3 = compress(images3)
    shuffle(images3)
    y3 = np.array(one_hots(3, 6))
    #images2 = x_y(images3, y3)
    image3_test  = images3[0:10]
    image3_val   = images3[10:20]
    image3_train = images3[20:]
    image3_test_target = [y3] * 10
    image3_val_target = [y3] * 10
    image3_train_target = [y3] * len(image3_train)
    
    ## GET IMAGES AND TARGETS FOR IMAGE 4
    images4 = get_all_images(CROPPED_PATH + sep + FOUR)
    images4 = compress(images4)
    shuffle(images4)
    y4 = np.array(one_hots(4, 6))
    #images4 = x_y(images4, y4)
    image4_test  = images4[0:10]
    image4_val   = images4[10:20]
    image4_train = images4[20:]
    image4_test_target = [y4] * 10
    image4_val_target = [y4] * 10
    image4_train_target = [y4] * len(image4_train)

    ## GET IMAGES AND TARGETS FOR IMAGE 5
    images5 = get_all_images(CROPPED_PATH + sep + FIVE)
    images5 = compress(images5)
    shuffle(images5)
    y5 = np.array(one_hots(5, 6))
    #images5 = x_y(images5, y5)
    image5_test  = images5[0:10]
    image5_val   = images5[10:20]
    image5_train = images5[20:]
    image5_test_target = [y5] * 10
    image5_val_target = [y5] * 10
    image5_train_target = [y5] * len(image5_train)

    ## GET IMAGES AND TARGETS FOR IMAGE 6
    images6 = get_all_images(CROPPED_PATH + sep + SIX)
    images6 = compress(images6)
    shuffle(images6)
    y6 = np.array(one_hots(6, 6))
    #images6 = x_y(images6, y6)
    image6_test  = images6[0:10]
    image6_val   = images6[10:20]
    image6_train = images6[20:]
    image6_test_target = [y6] * 10
    image6_val_target = [y6] * 10
    image6_train_target = [y6] * len(image6_train)
    

    #plt.imshow(decompress(image6_test)[0])
    #plt.show()
    
    training_images = big_data(image1_train, image2_train, image3_train, image4_train, image5_train, image6_train)
    test_images     = big_data(image1_test, image2_test, image3_test, image4_test, image5_test, image6_test)
    val_images      = big_data(image1_val, image2_val, image3_val, image4_val, image5_val, image6_val)

    training_images_targets = big_data(image1_train_target, image2_train_target, image3_train_target, image4_train_target, image5_train_target, image6_train_target)
    test_images_targets     = big_data(image1_test_target, image2_test_target, image3_test_target, image4_test_target, image5_test_target, image6_test_target)
    val_images_targets      = big_data(image1_val_target, image2_val_target, image3_val_target, image4_val_target, image5_val_target, image6_val_target)
    
    # This is for testing
    inputs_train = big_data(image1_train, image2_train, image3_train, image4_train, image5_train, image6_train)
    inputs_test     = big_data(image1_test, image2_test, image3_test, image4_test, image5_test, image6_test)
    inputs_valid      = big_data(image1_val, image2_val, image3_val, image4_val, image5_val, image6_val)
    
    
    target_train = big_data(image1_train_target, image2_train_target, image3_train_target, image4_train_target, image5_train_target, image6_train_target)
    target_test     = big_data(image1_test_target, image2_test_target, image3_test_target, image4_test_target, image5_test_target, image6_test_target)
    target_valid      = big_data(image1_val_target, image2_val_target, image3_val_target, image4_val_target, image5_val_target, image6_val_target)

    """
    c_training = list(zip(training_images, training_images_targets))
    c_test     = list(zip(test_images, test_images_targets))
    c_val      = list(zip(val_images, val_images_targets))

    random.shuffle(c_training)
    random.shuffle(c_test)
    random.shuffle(c_val)

    inputs_train, target_train = zip(*c_training)
    inputs_test, target_test = zip(*c_test)
    inputs_valid, target_valid = zip(*c_val)
    """
    
    inputs_train = np.array(inputs_train)
    inputs_test  = np.array(inputs_test)
    inputs_valid = np.array(inputs_valid)

    target_train = np.array(target_train)
    target_test  = np.array(target_test)
    target_valid = np.array(target_valid)

    #print(inputs_train.shape)
    #print(target_train)

    #inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('../toronto_face.npz')
    #print(target_train.shape)
    #raise(Exception, "Stop")
    rnd_idx = np.arange(inputs_train.shape[0])

    train_ce_list = []
    valid_ce_list = []
    train_acc_list = []
    valid_acc_list = []
    num_train_cases = inputs_train.shape[0]
    print("NUM TRAIN CASES", num_train_cases)
    if batch_size == -1:
        batch_size = num_train_cases
    num_steps = int(np.ceil(num_train_cases / batch_size))
    print("NUM STEPS", num_steps)

    for epoch in range(num_epochs):
        np.random.shuffle(rnd_idx)
        #print(inputs_train.shape)
        #raise(Expection, "Stop")
        inputs_train = inputs_train[rnd_idx]
        target_train = target_train[rnd_idx]
        for step in range(num_steps):
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            x = inputs_train[start: end]
            t = target_train[start: end]
            var = forward(model, x)
            
        
            prediction = Softmax(var['y'])
            train_ce = -np.sum(t * np.log(prediction)) / x.shape[0]
            train_acc = (np.argmax(prediction, axis=1) ==
                         np.argmax(t, axis=1)).astype('float').mean()
            print(('Epoch {:3d} Step {:2d} Train CE {:.5f} '
                   'Train Acc {:.5f}').format(
                epoch, step, train_ce, train_acc))

            # Compute error.
            error = (prediction - t) / x.shape[0]

            # Backward prop.
            backward(model, error, var)

            # Update weights.
            update(model, eps, momentum)

        valid_ce, valid_acc = Evaluate(
            inputs_valid, target_valid, model, forward, batch_size=batch_size)
        print(('Epoch {:3d} '
               'Validation CE {:.5f} '
               'Validation Acc {:.5f}\n').format(
            epoch, valid_ce, valid_acc))
        train_ce_list.append((epoch, train_ce))
        train_acc_list.append((epoch, train_acc))
        valid_ce_list.append((epoch, valid_ce))
        valid_acc_list.append((epoch, valid_acc))
        DisplayPlot(train_ce_list, valid_ce_list, 'Cross Entropy', number=0)
        DisplayPlot(train_acc_list, valid_acc_list, 'Accuracy', number=1)
        if epoch == (num_epochs - 1):
            ce_pic = DisplayPlot(train_ce_list, valid_ce_list, 'Cross Entropy', number=0)
            savefig('ce_pic.png')
            ac_pic = DisplayPlot(train_acc_list, valid_acc_list, 'Accuracy', number=1)
            savefig('ac_pic.png')


    print()
    train_ce, train_acc = Evaluate(
        inputs_train, target_train, model, forward, batch_size=batch_size)
    valid_ce, valid_acc = Evaluate(
        inputs_valid, target_valid, model, forward, batch_size=batch_size)
    test_ce, test_acc = Evaluate(
        inputs_test, target_test, model, forward, batch_size=batch_size)
    print('CE: Train %.5f Validation %.5f Test %.5f' %
          (train_ce, valid_ce, test_ce))
    print('Acc: Train {:.5f} Validation {:.5f} Test {:.5f}'.format(
        train_acc, valid_acc, test_acc))

    stats = {
        'train_ce': train_ce_list,
        'valid_ce': valid_ce_list,
        'train_acc': train_acc_list,
        'valid_acc': valid_acc_list
    }

    return model, stats




def Evaluate(inputs, target, model, forward, batch_size=-1):
    """Evaluates the model on inputs and target.

    Args:
        inputs: Inputs to the network.
        target: Target of the inputs.
        model:  Dictionary of network weights.
    """
    num_cases = inputs.shape[0]
    if batch_size == -1:
        batch_size = num_cases
    num_steps = int(np.ceil(num_cases / batch_size))
    ce = 0.0
    acc = 0.0
    for step in range(num_steps):
        start = step * batch_size
        end = min(num_cases, (step + 1) * batch_size)
        x = inputs[start: end]
        t = target[start: end]
        prediction = Softmax(forward(model, x)['y'])
        ce += -np.sum(t * np.log(prediction))
        acc += (np.argmax(prediction, axis=1) == np.argmax(
            t, axis=1)).astype('float').sum()
    ce /= num_cases
    acc /= num_cases
    return ce, acc


def CheckGrad(model, forward, backward, name, x):
    """Check the gradients

    Args:
        model: Dictionary of network weights.
        name: Weights name to check.
        x: Fake input.
    """
    np.random.seed(0)
    var = forward(model, x)
    loss = lambda y: 0.5 * (y ** 2).sum()
    grad_y = var['y']
    backward(model, grad_y, var)
    grad_w = model['dE_d' + name].ravel()
    w_ = model[name].ravel()
    eps = 1e-7
    grad_w_2 = np.zeros(w_.shape)
    check_elem = np.arange(w_.size)
    np.random.shuffle(check_elem)
    # Randomly check 20 elements.
    check_elem = check_elem[:20]
    for ii in check_elem:
        w_[ii] += eps
        err_plus = loss(forward(model, x)['y'])
        w_[ii] -= 2 * eps
        err_minus = loss(forward(model, x)['y'])
        w_[ii] += eps
        grad_w_2[ii] = (err_plus - err_minus) / 2 / eps
    np.testing.assert_almost_equal(grad_w[check_elem], grad_w_2[check_elem], decimal=3)


def main(file_name=None):
    """Trains a NN."""
    if file_name is None:
        #model_fname = 'nn_model.npz'
        #stats_fname = 'nn_stats.npz'

        # Hyper-parameters. Modify them if needed.
        l_eps = [0.001, 0.01, 0.1, 0.5, 1.0]
        l_momentum = [0, 0.5, 0.9]
        l_batch_size = [10, 100, 500, 700, 1000]
        l_num_hiddens = [[2, 6], [16, 32], [70, 100]]
        
        # default
        num_hiddens = [16, 32]
        eps = 0.0001
        momentum = 0.01
        num_epochs = 1200
        batch_size = 100

        # Input-output dimensions.
        num_inputs = X_DIM
        num_outputs = Y_DIM

        # Initialize model.
        model = InitNN(num_inputs, num_hiddens, num_outputs)

        # Uncomment to reload trained model here.
        # model = Load(model_fname)

        # Train model.
        stats = Train(model, NNForward, NNBackward, NNUpdate, eps,
                      momentum, num_epochs, batch_size)
    
    if file_name is not None:
        ## After training, show me some images where the Network miss classifies
        ## Using batch size of 1 on trained model.
        #inputs_train, inputs_valid, inputs_test, target_train, target_valid, \
        #target_test = LoadData('../toronto_face.npz')

        #with open('<FILENAME>') as f:
        #    model['W1'] = cPickle.load(f), name="model['W1']")
        
        model = Load('model_fname.npz')
        # print(model)
        
        #print("TYPE" , model)
        #print("MODEL: ", model['dE_db1'])
        var = NNForward(model, file_name)
        
        #prediction = Softmax(var['y'])
        #print(prediction)

        i = 0
        for count in range(6):
            if var['y'][count] > var['y'][i]:
                i = count

        print(i)

    """
    batch_size = 1
    rnd_idx = np.arange(inputs_test.shape[0])
    num_test_cases = inputs_test.shape[0]
    num_steps = int(np.ceil(num_test_cases / batch_size))
    for epoch in range(num_epochs):
        np.random.shuffle(rnd_idx)
        inputs_test = inputs_test[rnd_idx]
        target_test = target_test[rnd_idx]
        for step in range(num_steps):
            # Forward prop.
            start = step * batch_size
            end = min(num_test_cases, (step + 1) * batch_size)
            x = inputs_test[start: end]
            t = target_test[start: end]

            var = NNForward(model, x)
            prediction = Softmax(var['y'])

            train_ce = -np.sum(t * np.log(prediction)) / x.shape[0]
            train_acc = (np.argmax(prediction, axis=1) ==
                         np.argmax(t, axis=1)).astype('float').mean()
            print(('Epoch {:3d} Step {:2d} Test CE {:.5f} '
                   'Test Acc {:.5f}').format(
                epoch, step, train_ce, train_acc))
            if train_acc < 0.5:
                plt.clf()
                plt.imshow(x[0].reshape(48, 48), cmap=plt.cm.gray)
                plt.draw()
                print(train_acc)
                raw_input('Press Enter.')
    """
    model_fname = 'model_fname.npz'
    stats_fname = 'stats_fname.npz'
    # Uncomment if you wish to save the model.
    Save(model_fname, dict(model))

    # Uncomment if you wish to save the training statistics.
    # Save(stats_fname, dict(stats))

    """
    with open('<FILENAME>') as f:
        cPickle.dump(model['W1'].get_value(), f, pickle.HIGHEST_PROTOCOL)
        cPickle.dump(model['W2'].get_value(), f, pickle.HIGHEST_PROTOCOL)
        cPickle.dump(model['W3'].get_value(), f, pickle.HIGHEST_PROTOCOL)
        cPickle.dump(model['b1'].get_value(), f, pickle.HIGHEST_PROTOCOL)
        cPickle.dump(model['b2'].get_value(), f, pickle.HIGHEST_PROTOCOL)
        cPickle.dump(model['b3'].get_value(), f, pickle.HIGHEST_PROTOCOL)
    """


if __name__ == '__main__':
    args = sys.argv
    
    if len(args) < 2:
        main()
    else:
        print(args[1])
        file_path = args[1]
        img = imread(file_path).ravel()
        main(img)



"""
    
                __                             ___            _aaaa
               d8888aa,_                    a8888888a   __a88888888b
              d8P   `Y88ba.                a8P'~~~~Y88a888P""~~~~Y88b
             d8P      ~"Y88a____aaaaa_____a8P        888          Y88
            d8P          ~Y88"8~~~~~~~88888P          88g          88
           d8P                           88      ____ _88y__       88b
           88                           a88    _a88~8888"8M88a_____888
           88                           88P    88  a8"'     `888888888b_
          a8P                           88     88 a88         88b     Y8,
           8b                           88      8888P         388      88b
          a88a                          Y8b       88L         8888.    88P
         a8P                             Y8_     _888       _a8P 88   a88
        _8P                               ~Y88a888~888g_   a888yg8'  a88'
        88                                   ~~~~    ~""8888        a88P
       d8'                                                Y8,      888L
       8E                                                  88a___a8"888
      d8P                                                   ~Y888"   88L
      88                                                      ~~      88
      88                                                              88
      88                                                              88b
  ____88a_.      a8a                                                __881
88""P~888        888b                                 __          8888888888
      888        888P                                d88b             88
     _888ba       ~            aaaa.                 8888            d8P
 a888~"Y88                    888888                 "8P          8aa888_
        Y8b                   Y888P"                                88""888a
        _88g8                  ~~~                                 a88    ~~
    __a8"888_                                                  a_ a88
   88"'    "88g                                                 "888g_
   ~         `88a_                                            _a88'"Y88gg,
                "888aa_.                                   _a88"'      ~88
                   ~~""8888aaa______                ____a888P'
                           ~~""""""888888888888888888""~~~
                                      ~~~~~~~~~~~~



"""
