import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#%matplotlib inline
import random, os
from os.path import sep

from scipy.misc import imread
from scipy.misc import imresize
from random import shuffle


"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Generate predetermined random weights so the networks are similarly initialized
w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

# Small epsilon value for the BN transform
epsilon = 1e-3

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Layer 1 without BN
w1 = tf.Variable(w1_initial)
b1 = tf.Variable(tf.zeros([100]))
z1 = tf.matmul(x,w1)+b1
l1 = tf.nn.sigmoid(z1)

# Layer 1 with BN
w1_BN = tf.Variable(w1_initial)

# Note that pre-batch normalization bias is ommitted. The effect of this bias would be
# eliminated when subtracting the batch mean. Instead, the role of the bias is performed
# by the new beta variable. See Section 3.2 of the BN2015 paper.
z1_BN = tf.matmul(x,w1_BN)

# Calculate batch mean and variance
batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])

# Apply the initial batch normalizing transform
z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

# Create two new parameters, scale and beta (shift)
scale1 = tf.Variable(tf.ones([100]))
beta1 = tf.Variable(tf.zeros([100]))

# Scale and shift to obtain the final output of the batch normalization
# this value is fed into the activation function (here a sigmoid)
BN1 = scale1 * z1_hat + beta1
l1_BN = tf.nn.sigmoid(BN1)

# Layer 2 without BN
w2 = tf.Variable(w2_initial)
b2 = tf.Variable(tf.zeros([100]))
z2 = tf.matmul(l1,w2)+b2
l2 = tf.nn.sigmoid(z2)

# Layer 2 with BN, using Tensorflows built-in BN function
w2_BN = tf.Variable(w2_initial)
z2_BN = tf.matmul(l1_BN,w2_BN)
batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
z2_hat = (z2_BN - batch_mean2) / tf.sqrt(batch_var2 + epsilon)
scale2 = tf.Variable(tf.ones([100]))
beta2 = tf.Variable(tf.zeros([100]))
BN2 = scale2 * z2_hat + beta2
l2_BN = tf.nn.sigmoid(BN2)

# Softmax
w3 = tf.Variable(w3_initial)
b3 = tf.Variable(tf.zeros([10]))
y  = tf.nn.softmax(tf.matmul(l2,w3)+b3)

w3_BN = tf.Variable(w3_initial)
b3_BN = tf.Variable(tf.zeros([10]))
y_BN  = tf.nn.softmax(tf.matmul(l2_BN,w3_BN)+b3_BN)

# Loss, optimizer and predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step_BN = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_BN)

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
correct_prediction_BN = tf.equal(tf.arg_max(y_BN,1),tf.arg_max(y_,1))
accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN,tf.float32))
zs, BNs, acc, acc_BN = [], [], [], []

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(10000):
    batch = mnist.train.next_batch(60)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 50 is 0:
        res = sess.run([accuracy,accuracy_BN,z2,BN2],
          feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        acc.append(res[0])
        acc_BN.append(res[1])
        zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
        BNs.append(np.mean(res[3],axis=0)) # record the mean value of BN2 over the entire test set
        #print(".",end="")
        print(str((i/10000.0)*100) + "% done")

zs, BNs, acc, acc_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN)

fig, ax = plt.subplots()

ax.plot(range(0,len(acc)*50,50),acc, label='Without BN')
ax.plot(range(0,len(acc)*50,50),acc_BN, label='With BN')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.8,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.show()

"""
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
BATCH_SIZE = 100
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
    for thing in images1:
        big_data.append(thing)
    for thing in images2:
        big_data.append(thing)
    for thing in images3:
        big_data.append(thing)
    for thing in images4:
        big_data.append(thing)
    for thing in images5:
        big_data.append(thing)
    for thing in images6:
        big_data.append(thing)
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

#print x_y([[1,2], [3,4]], 7)
#print one_hots(1, 6)
#print one_hots(6, 6)

images1 = get_all_images(CROPPED_PATH + sep + ONE)
shuffle(images1)
y1 = one_hots(1, 6)
x_y(images1, y1)
image1_test  = images1[0:10]
image1_val   = images1[10:20]
image1_train = images1[20:]

images2 = get_all_images(CROPPED_PATH + sep + TWO)
shuffle(images2)
y2 = one_hots(2, 6)
x_y(images2, y2)
image2_test  = images2[0:10]
image2_val   = images2[10:20]
image2_train = images2[20:]

images3 = get_all_images(CROPPED_PATH + sep + THREE)
shuffle(images3)
y3 = one_hots(3, 6)
x_y(images3, y3)
image3_test  = images3[0:10]
image3_val   = images3[10:20]
image3_train = images3[20:]

images4 = get_all_images(CROPPED_PATH + sep + FOUR)
shuffle(images4)
y4 = one_hots(4, 6)
x_y(images4, y4)
image4_test  = images4[0:10]
image4_val   = images4[10:20]
image4_train = images4[20:]

images5 = get_all_images(CROPPED_PATH + sep + FIVE)
shuffle(images5)
y5 = one_hots(5, 6)
x_y(images5, y5)
image5_test  = images5[0:10]
image5_val   = images5[10:20]
image5_train = images5[20:]

images6 = get_all_images(CROPPED_PATH + sep + SIX)
shuffle(images6)
y6 = one_hots(6, 6)
x_y(images6, y6)
image6_test  = images6[0:10]
image6_val   = images6[10:20]
image6_train = images6[20:]

training_images = big_data(image1_train, image2_train, image3_train, image4_train, image5_train, image6_train)
test_images     = big_data(image1_test, image2_test, image3_test, image4_test, image5_test, image6_test)
val_images      = big_data(image1_val, image2_val, image3_val, image4_val, image5_val, image6_val)

train_inputs, train_targets = uncouple(training_images)
val_inputs, val_targets = uncouple(val_images)
test_inputs, test_targets = uncouple(test_images)

num_train_cases = len(training_images)


#num_batches = len(training_images) / BATCH_SIZE
batches = []
for idx in range(0, num_train_cases, BATCH_SIZE):
    batch = training_images[idx : idx + batch_size]
    batches.append(batch)

#plt.imshow(training_images[0])
#plt.show()

training_images = compress(training_images)
test_images     = compress(test_images)
val_images      = compress(val_images)

#plt.imshow(decompress(training_images)[0])
#plt.show()

# Generate predetermined random weights so the networks are similarly initialized
w1_initial = np.random.normal(size=(X_DIM,100)).astype(np.float32)
w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
w3_initial = np.random.normal(size=(100,Y_DIM)).astype(np.float32)

# Small epsilon value for the BN transform
epsilon = 1e-3

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, X_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, Y_DIM])

# Layer 1 without BN
w1 = tf.Variable(w1_initial)
b1 = tf.Variable(tf.zeros([100]))
z1 = tf.matmul(x,w1)+b1
l1 = tf.nn.sigmoid(z1)

# Layer 1 with BN
w1_BN = tf.Variable(w1_initial)

# Note that pre-batch normalization bias is ommitted. The effect of this bias would be
# eliminated when subtracting the batch mean. Instead, the role of the bias is performed
# by the new beta variable. See Section 3.2 of the BN2015 paper.
z1_BN = tf.matmul(x,w1_BN)

# Calculate batch mean and variance
batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])

# Apply the initial batch normalizing transform
z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

# Create two new parameters, scale and beta (shift)
scale1 = tf.Variable(tf.ones([100]))
beta1 = tf.Variable(tf.zeros([100]))

# Scale and shift to obtain the final output of the batch normalization
# this value is fed into the activation function (here a sigmoid)
BN1 = scale1 * z1_hat + beta1
l1_BN = tf.nn.sigmoid(BN1)

# Layer 2 without BN
w2 = tf.Variable(w2_initial)
b2 = tf.Variable(tf.zeros([100]))
z2 = tf.matmul(l1,w2)+b2
l2 = tf.nn.sigmoid(z2)

# Layer 2 with BN, using Tensorflows built-in BN function
w2_BN = tf.Variable(w2_initial)
z2_BN = tf.matmul(l1_BN,w2_BN)
batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
z2_hat = (z2_BN - batch_mean2) / tf.sqrt(batch_var2 + epsilon)
scale2 = tf.Variable(tf.ones([100]))
beta2 = tf.Variable(tf.zeros([100]))
BN2 = scale2 * z2_hat + beta2
l2_BN = tf.nn.sigmoid(BN2)

# Softmax
w3 = tf.Variable(w3_initial)
b3 = tf.Variable(tf.zeros([10]))
y  = tf.nn.softmax(tf.matmul(l2,w3)+b3)

w3_BN = tf.Variable(w3_initial)
b3_BN = tf.Variable(tf.zeros([10]))
y_BN  = tf.nn.softmax(tf.matmul(l2_BN,w3_BN)+b3_BN)

# Loss, optimizer and predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
train_step_BN = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_BN)

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
correct_prediction_BN = tf.equal(tf.arg_max(y_BN,1),tf.arg_max(y_,1))
accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN,tf.float32))
zs, BNs, acc, acc_BN = [], [], [], []

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(NUM_EPOCHS):
    for batch in batches:
        #batch = mnist.train.next_batch(60)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 50 is 0:
            res = sess.run([accuracy,accuracy_BN,z2,BN2],
              feed_dict={x: test_inputs, y_: test_targets})
            acc.append(res[0])
            acc_BN.append(res[1])
            zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
            BNs.append(np.mean(res[3],axis=0)) # record the mean value of BN2 over the entire test set
            #print(".",end="")
            print(str((i/10000.0)*100) + "% done")


    res = sess.run([accuracy_BN, BN2],
      feed_dict={x: valid_batch[0], y_: valid_batch[1]})
    #acc.append(res[0])
    acc_BN.append(res[0])
    # zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
    BNs.append(np.mean(res[1],axis=0)) # record the mean value of BN2 over the entire test set
    #print(".",end="")


zs, BNs, acc, acc_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN)

fig, ax = plt.subplots()

ax.plot(range(0,len(acc)*50,50),acc, label='Without BN')
ax.plot(range(0,len(acc)*50,50),acc_BN, label='With BN')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.8,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.show()


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



"""
inputs = np.load("cropped/Angie \Harmon")
inputs = compress(inputs)

valids = np.load("Val_Images.npy")
valids = compress(inputs)




valids = inputs[5600: 7001]
inputs = inputs[0   :5600]

reader = np.loadtxt(open("train_mod.csv","rb"),delimiter=",",skiprows=1)
woo = list(reader)
result = np.array(woo).astype('int')
targets = [t[1] for t in result]
targets = np.array(targets)
num_train_cases = inputs.shape[0]

hot_targets = []
for tar in range(len(targets)):
    #print targets[tar]
    one_hot = [0, 0, 0, 0, 0, 0, 0, 0]
    one_hot[targets[tar] - 1] = 1
    hot_targets.append(one_hot)


hot_targets = np.array(hot_targets)

valid_targets = hot_targets[5600: 7001]
hot_targets   = hot_targets[0   :5600]

### HYPERPARAMETERS ###

num_epochs = 50
batch_size = 500

#######################

inputs = np.array([(inputs[i895], hot_targets[i895])   for i895 in range(len(inputs))])
valids = np.array([(valids[i885], valid_targets[i885]) for i885 in range(len(valids))])

lo = [inputs[q][0] for q in range(len(valids))]
la = [inputs[p][1] for p in range(len(valids))]
valid_batch = (lo, la)


# Generate predetermined random weights so the networks are similarly initialized
w1_initial = np.random.normal(size=(X_DIM,100)).astype(np.float32)
w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
w3_initial = np.random.normal(size=(100,Y_DIM)).astype(np.float32)

#show_weights = w1_initial


# Small epsilon value for the BN transform
epsilon = 1e-3

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, X_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, Y_DIM])

# Layer 1 with BN
w1_BN = tf.Variable(w1_initial)

# Note that pre-batch normalization bias is ommitted. The effect of this bias would be
# eliminated when subtracting the batch mean. Instead, the role of the bias is performed
# by the new beta variable. See Section 3.2 of the BN2015 paper.
z1_BN = tf.matmul(x,w1_BN)

# Calculate batch mean and variance
batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])

# Apply the initial batch normalizing transform
z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

# Create two new parameters, scale and beta (shift)
scale1 = tf.Variable(tf.ones([100]))
beta1 = tf.Variable(tf.zeros([100]))

# Scale and shift to obtain the final output of the batch normalization
# this value is fed into the activation function (here a sigmoid)
BN1 = scale1 * z1_hat + beta1
l1_BN = tf.nn.sigmoid(BN1)

# Layer 2 with BN
w2_BN = tf.Variable(w2_initial)
z2_BN = tf.matmul(l1_BN,w2_BN)
batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
z2_hat = (z2_BN - batch_mean2) / tf.sqrt(batch_var2 + epsilon)
scale2 = tf.Variable(tf.ones([100]))
beta2 = tf.Variable(tf.zeros([100]))
BN2 = scale2 * z2_hat + beta2
l2_BN = tf.nn.sigmoid(BN2)

w3_BN = tf.Variable(w3_initial)
b3_BN = tf.Variable(tf.zeros([Y_DIM]))
y_BN  = tf.nn.softmax(tf.matmul(l2_BN,w3_BN)+b3_BN)

# Loss, optimizer and predictions
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))

#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step_BN = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_BN)

#correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
correct_prediction_BN = tf.equal(tf.arg_max(y_BN,1),tf.arg_max(y_,1))
accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN,tf.float32))
#zs, BNs, acc, acc_BN = [], [], [], []
BNs, acc_BN = [], []

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(num_epochs):
    print str(i/float(batch_size)) + "% done"
    for idx in range(0, num_train_cases, batch_size):
        print idx
        # change it to a touple: (all inputs, all targets)
        batch = inputs[idx : idx + batch_size]

        
        a = [inputs[u][0] for u in range(len(batch))]
        b = [inputs[o][1] for o in range(len(batch))]
        batch = (a, b)

        #train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
        #acc_BN.append(res[1])
        #zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
        #BNs.append(np.mean(res[3],axis=0)) # record the mean value of BN2 over the entire test set
        #print(".",end="")
        #print(str((i/10000.0)*100) + "% done")
        
        
    res = sess.run([accuracy_BN, BN2],
      feed_dict={x: valid_batch[0], y_: valid_batch[1]})
    #acc.append(res[0])
    acc_BN.append(res[0])
    # zs.append(np.mean(res[2],axis=0)) # record the mean value of z2 over the entire test set
    BNs.append(np.mean(res[1],axis=0)) # record the mean value of BN2 over the entire test set
    #print(".",end="")

            



print acc_BN

BNs, acc_BN = np.array(BNs), np.array(acc_BN)

fig, ax = plt.subplots()


#.plot(range(0,len(acc_BN)*50,50),acc, label='Without BN')
ax.plot(range(0,len(acc_BN)*50,50),acc_BN, label='With BN')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.show()


"""











