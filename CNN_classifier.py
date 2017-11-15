### Shervin Minaee, July 2017
### Ad categorization using CNN on video key-frames
### 10 classes, 12k raw data. 
### Data augmentation is applied, by flipping, rotating, and adding noise

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle, os, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import linalg as LA


init_time= time.time()


#################################### Loading the Dataset
ads_data_list= pickle.load( open("Ads400_3keyframes_july19.p", "rb") )
ads_data= np.asarray(ads_data_list) ## 12k samples of 112x112x3
vid_data_list= pickle.load( open("Vid500_10shots_3keyframes.p", "rb") )
vid_data= np.asarray(vid_data_list) ## 12k samples of 112x112x3
ads_audio_list= pickle.load( open("Ads400_audio_ad_mfcc_july19.p", "rb") )
ads_audio= np.asarray(ads_audio_list) ## 12k samples of 112x112x3
vid_audio_list= pickle.load( open("Vid500_10shots_mfcc.p", "rb") )
vid_audio= np.asarray(vid_audio_list) ## 12k samples of 112x112x3


################################### Audio normalization and repeating
for i in range(ads_audio.shape[0]):
    ads_audio[i,:]= ( ads_audio[i,:]- np.mean(ads_audio[i,:]) ) / ( np.std(ads_audio[i,:])+ 1e-6 )
for i in range(vid_audio.shape[0]):
    vid_audio[i,:]= ( vid_audio[i,:]- np.mean(vid_audio[i,:]) ) / ( np.std(vid_audio[i,:])+ 1e-6 )
     
ads_audio_3x= np.zeros([ads_audio.shape[0]*3, ads_audio.shape[1]])
vid_audio_3x= np.zeros([vid_audio.shape[0]*3, vid_audio.shape[1]])
for i in range(ads_audio.shape[0]):
    ads_audio_3x[3*i,:]=   ads_audio[i,:]
    ads_audio_3x[3*i+1,:]= ads_audio[i,:]
    ads_audio_3x[3*i+2,:]= ads_audio[i,:]

for i in range(vid_audio.shape[0]):
    vid_audio_3x[3*i,:]=   vid_audio[i,:]
    vid_audio_3x[3*i+1,:]= vid_audio[i,:]
    vid_audio_3x[3*i+2,:]= vid_audio[i,:]


#################################### Visual part normalizatin
print("\n")
zero_std_ads= list()
for i in range(ads_data.shape[0]):
    for j in range(ads_data.shape[3]):
        if (np.std(ads_data[i,:,:,j])< 1e-6):
            zero_std_ads.append(i)
            ads_data[i,:,:,j]= ( ads_data[i,:,:,j]- np.mean(ads_data[i,:,:,j]) ) / ( LA.norm(ads_data[i,:,:,j])+ 1e-6 )
        else:
            ads_data[i,:,:,j]= ( ads_data[i,:,:,j]- np.mean(ads_data[i,:,:,j]) ) / ( np.std(ads_data[i,:,:,j]) + 1e-8 )              
    if i%1000==0:
        print("%d-th ads' sample normalized!" %i)
   
print("\n")
zero_std_vid= list()
for i in range(vid_data.shape[0]):
    for j in range(vid_data.shape[3]):
        if (np.std(vid_data[i,:,:,j])< 1e-6):
            zero_std_vid.append(i)
            vid_data[i,:,:,j]= ( vid_data[i,:,:,j]- np.mean(vid_data[i,:,:,j]) ) / ( LA.norm(vid_data[i,:,:,j])+ 1e-6 )
        else:
            vid_data[i,:,:,j]= ( vid_data[i,:,:,j]- np.mean(vid_data[i,:,:,j]) ) / ( np.std(vid_data[i,:,:,j]) + 1e-8 )              
    if i%1000==0:
        print("%d-th video sample normalized!" %i)      
        

#################################### shuffle ads and videos, to have from all classes in training
num_frame_pv= 30 ## number of frames per video
num_ads_shot= int(ads_data.shape[0]/30)
num_vid_shot= int(vid_data.shape[0]/30)
np.random.seed(0)
shuffled_ind1= np.random.permutation( num_ads_shot)
shuffled_ind2= np.random.permutation( num_vid_shot)

X1_shuffled_ads = np.zeros( ads_data.shape, dtype=np.float)      ## visual frame
X2_shuffled_ads = np.zeros( ads_audio_3x.shape, dtype=np.float)  ## audio mfcc features
for i in range( num_ads_shot):
    X1_shuffled_ads[ 30*i:30*i+30, :, :, :] = ads_data[ 30*shuffled_ind1[i]:30*shuffled_ind1[i]+30, :, :, :]
    X2_shuffled_ads[ 30*i:30*i+30, :] =   ads_audio_3x[ 30*shuffled_ind1[i]:30*shuffled_ind1[i]+30, :]
    
X1_shuffled_vid= np.zeros( vid_data.shape, dtype=np.float)
X2_shuffled_vid= np.zeros( vid_audio_3x.shape, dtype=np.float)  ## audio mfcc features
for i in range( num_vid_shot):
    X1_shuffled_vid[ 30*i:30*i+30, :, :, :] = vid_data[ 30*shuffled_ind2[i]:30*shuffled_ind2[i]+30, :, :, :]
    X2_shuffled_vid[ 30*i:30*i+30, :] =   vid_audio_3x[ 30*shuffled_ind2[i]:30*shuffled_ind2[i]+30, :]

training_percentage= 0.8
num_ads_train= int( training_percentage*ads_data.shape[0])
num_ads_test = ads_data.shape[0]- num_ads_train
num_vid_train= int( training_percentage*vid_data.shape[0])
num_vid_test = vid_data.shape[0]- num_vid_train

X1_train_ads= X1_shuffled_ads[:num_ads_train,:,:,:]
X1_test_ads = X1_shuffled_ads[num_ads_train:,:,:,:]
X1_train_vid= X1_shuffled_vid[:num_vid_train,:,:,:]
X1_test_vid = X1_shuffled_vid[num_vid_train:,:,:,:]

X2_train_ads= X2_shuffled_ads[:num_ads_train,:]
X2_test_ads = X2_shuffled_ads[num_ads_train:,:]
X2_train_vid= X2_shuffled_vid[:num_vid_train,:]
X2_test_vid = X2_shuffled_vid[num_vid_train:,:]


#### preparing the labels: ads= [1,0] and videos=[0,1]
Y_train_ads= np.hstack( ( np.ones((num_ads_train,1), dtype=np.int8), np.zeros((num_ads_train,1), dtype=np.int8) ))
Y_train_vid= np.hstack( ( np.zeros((num_vid_train,1), dtype=np.int8), np.ones((num_vid_train,1), dtype=np.int8) ))
Y_test_ads = np.hstack( ( np.ones((num_ads_test,1), dtype=np.int8), np.zeros((num_ads_test,1), dtype=np.int8) ))
Y_test_vid = np.hstack( ( np.zeros((num_vid_test,1), dtype=np.int8), np.ones((num_vid_test,1), dtype=np.int8) ))

X1_train= np.vstack( (X1_train_ads, X1_train_vid) )
X1_test = np.vstack( (X1_test_ads,  X1_test_vid) )
X2_train= np.vstack( (X2_train_ads, X2_train_vid) )
X2_test = np.vstack( (X2_test_ads,  X2_test_vid) )
Y_train = np.vstack( (Y_train_ads, Y_train_vid) )
Y_test  = np.vstack( (Y_test_ads,  Y_test_vid) )

np.random.seed(1)
shuffled_ind4= np.random.permutation( X1_test.shape[0])
X1_test= X1_test[shuffled_ind4,:,:,:]
X2_test= X2_test[shuffled_ind4,:]
Y_test = Y_test[shuffled_ind4,:]

np.random.seed(1)
shuffled_ind5= np.random.permutation( X1_train.shape[0])
X1_train1= X1_train[shuffled_ind5,:,:,:]
X2_train1= X2_train[shuffled_ind5,:]
Y_train1 = Y_train[shuffled_ind5,:]




print("Data processing is Done!")

pred_test_label= np.zeros(Y_test.shape)
#################################### Hyper-parameters
learning_rate= tf.placeholder( tf.float32, shape=[], name="learning_rate")
num_epoch= 100
batch_size= 200
num_batches= int( X1_train.shape[0]/batch_size )
num_classes= 2

train_acc_epoch= np.zeros((5*num_epoch,))
test_acc_epoch = np.zeros((5*num_epoch,))
test_pre_epoch = np.zeros((5*num_epoch,))
test_rec_epoch = np.zeros((5*num_epoch,))
test_f1s_epoch = np.zeros((5*num_epoch,))

#################################### TF Graph Input
x1 = tf.placeholder(tf.float32, [None, vid_data.shape[1], vid_data.shape[2], 3], name="video_frame")
x2 = tf.placeholder(tf.float32, [None, ads_audio_3x.shape[1]], name="MFCC_feature" )
y =  tf.placeholder(tf.float32, [None, num_classes], name="ad_label")  #### ground-truth class labels
keep_prob = tf.placeholder(tf.float32, name="dropout_rate") ## dropout (keep probability)

#################################### Network Architecture
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.03)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
  
def conv2d_st2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')




################################################# Visual Information Processing Network
# 1st Convolutional Layer kernel dim= (5x5x3), #output channels= 16. 
# Rectified linear output & max pooling. Output size= floor[(W1-K+2P)/S]+1 
W_conv1 = tf.Variable(tf.truncated_normal( [3, 3, 3, 16], stddev=0.03), name="w_conv1") ## 
b_conv1 = tf.Variable(tf.constant(0.01, shape=[16]), name="b_conv1")             
h_conv1 = tf.nn.relu(conv2d( x1, W_conv1) + b_conv1, "conv1") ## output size= 110x110x32
h_pool1 = max_pool_2x2(h_conv1)                          ## output= 55x55x32


# 2nd Convolutional Layer kernel dim= (3x3x16), #output channels= 20
W_conv2 = tf.Variable(tf.truncated_normal( [5, 5, 16, 10], stddev=0.03), name="w_conv2")  ## 
b_conv2 = tf.Variable(tf.constant(0.01, shape=[10]), name="b_conv2")
h_conv2 = tf.nn.relu(conv2d_st2(h_pool1, W_conv2) + b_conv2, "conv2") ## output size= 26x26x16
h_pool2 = max_pool_2x2(h_conv2)                          ## output= 13x13x20


# 3rd Convolutional Layer kernel dim= (3x3x20), #output channels= 32
W_conv3 = tf.Variable(tf.truncated_normal( [3, 3, 10, 10], stddev=0.03), name="w_conv3")  ## 
b_conv3 = tf.Variable(tf.constant(0.01, shape=[10]), name="b_conv3")
h_conv3 = tf.nn.relu(conv2d_st2(h_pool2, W_conv3) + b_conv3, "conv3") ## output size= 6x6x10
h_pool3 = max_pool_2x2(h_conv3)                          ## output= 3x3x10


# 1st fully connected layer. flatten last convolutional output, 50-dim
W_fc1 = tf.Variable(tf.truncated_normal( [3*3*10, 20], stddev=0.03), name="W_fc1")  ## 
b_fc1 = tf.Variable(tf.constant(0.01, shape=[20]), name="b_fc1")
h_pool3_flat = tf.reshape(h_pool3, [-1, 3*3*10])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, "fc1")



################### audio features processing
W_fc1_aud = tf.Variable(tf.truncated_normal( [200, 20], stddev=0.03), name="W_fc1_aud")  ## 
b_fc1_aud = tf.Variable(tf.constant(0.01, shape=[20]), name="b_fc1_aud")
h_fc1_aud = tf.nn.relu(tf.matmul( x2, W_fc1_aud) + b_fc1_aud, "aud_fc1")



################### fusing visual and audio information by concatenating them
# add dropout after fully connected layer
fused_video_audio= tf.concat([ h_fc1, h_fc1_aud], 1, "fused_feature")
h_fc1_drop = tf.nn.dropout( fused_video_audio, keep_prob )


# 2nd fully connected layer: output are the class probabilities
W_fc2 = tf.Variable(tf.truncated_normal( [40, num_classes], stddev=0.03), name="W_fc2")
b_fc2 = tf.Variable(tf.constant(0.01, shape=[num_classes]), name="b_fc2")
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="y_conv") ## predicted class proabilities: label



#### defining loss function and optimization
regularizers = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1)+ tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum( y*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)) ))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
cross_entropy = cross_entropy+ 1e-4*regularizers

optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cross_entropy)
#optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

################### precision recall computation
y_gr_1D= tf.argmax( y,axis=1)          ## 0 denotes ads, and 1 denotes videos
y_pr_1D= tf.argmax( y_conv, axis=1)    ## 0 denotes ads, 1 denotes videos
y_gr_1D= tf.cast( 1-y_gr_1D, tf.int64) ## 1 denotes ads, and 0 denotes videos
y_pr_1D= tf.cast( 1-y_pr_1D, tf.int64) ## 1 denotes ads, and 0 denotes videos
TP= tf.count_nonzero( y_gr_1D*y_pr_1D)
TN= tf.count_nonzero( (1-y_gr_1D)*(1-y_pr_1D) )
FP= tf.count_nonzero( (1-y_gr_1D)*y_pr_1D ) ##
FN= tf.count_nonzero( y_gr_1D*(1-y_pr_1D) )

#prec   = tf.divide( TP, TP+FP, name="precision")
#recall = tf.divide( TP, TP+FN, name="recall")
#f1     = tf.divide( 2*prec*recall, prec+recall, name="F1_score")
#acc_avg= tf.divide( (TP+TN), (TP+TN+FP+FN), name="acc_avg")
prec  = (TP)/(TP+FP)
recall= TP/(TP+FN)
f1= 2*prec*recall/(prec+recall)
acc_avg= (TP+TN)/(TP+TN+FP+FN)


################################### Initializing the variables
init = tf.global_variables_initializer()
display_step= 20
pool_size= 10
curr_test_label= np.zeros( (pool_size, num_classes) )
model_path= "./tmp/model.ckpt"
min_f1= 0.9
highest_rate_epoch= 0

saver= tf.train.Saver()


with tf.Session() as sess:
    sess.run(init)

    for ep_num in range(num_epoch):
        print("\n")
        shuffled_ind3= np.random.permutation( X1_train.shape[0])
        if ep_num==0:
            learn_rate= 0.0003  ##
        elif ep_num>= 10 and learn_rate> 0.00001:
            learn_rate= learn_rate*0.9
        else:
            learn_rate= learn_rate
        
        inner_iter= 0
        for j in range(num_batches):
        #for j in range(40):
        
            batch_x1 = X1_train[ shuffled_ind3[j*batch_size:(j+1)*batch_size], :, :, :]
            batch_x2 = X2_train[ shuffled_ind3[j*batch_size:(j+1)*batch_size], :]
            batch_y  = Y_train[ shuffled_ind3[j*batch_size:(j+1)*batch_size], :]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, learning_rate: learn_rate, keep_prob: 0.5 })
            
            if (j+1) % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x1: batch_x1, x2: batch_x2,
                                                              y: batch_y,
                                                              keep_prob: 1.})
                          
                print( str( ep_num+1)+"-th Epoch, " + str(j+1) + "-th Batch,"+ " Ent_Loss= " + \
                  "{:.3f}".format(loss) + ", Train Acc= " + \
                  "{:.3f}".format(acc))

                my_acc, my_prec, my_recall, my_f1 = sess.run([acc_avg, prec, recall, f1], feed_dict={x1: X1_test[:2000,:,:,:], x2: X2_test[:2000,:,],
                                                              y: Y_test[:2000,:], keep_prob: 1.})
                print("Test accuracy, precision, recall and F1= (%.3f,%.3f,%.3f,%.3f)\n" %(my_acc, my_prec, my_recall, my_f1) )               
                test_acc_epoch[5*ep_num+inner_iter]= my_acc
                test_pre_epoch[5*ep_num+inner_iter]= my_prec
                test_rec_epoch[5*ep_num+inner_iter]= my_recall
                test_f1s_epoch[5*ep_num+inner_iter]= my_f1

                train_acc1= sess.run( accuracy, feed_dict={x1: X1_train1[:2000,:,:,:], x2: X2_train1[:2000,:],
                                                              y: Y_train1[:2000,:],  keep_prob: 1.})
                train_acc_epoch[5*ep_num+inner_iter]= train_acc1
                inner_iter= inner_iter+1
                ####################### saving the model with highest accuracy
                if my_f1> min_f1:
                    min_f1= my_f1
                    #save_path= saver.save(sess, 'my_test_model30', global_step= 10)
                    print("model saved at epoch %d!" %(ep_num+1) )
                    highest_rate_epoch= ep_num
        
        
        
      
        if ep_num%10==0:
            for k in range(pool_size):
                curr_test_label[ k,:]= y_conv.eval( session=sess, feed_dict={ x1: np.reshape(X1_test[k,:,:,:],[1,X1_test.shape[1], X1_test.shape[2], X1_test.shape[3]]), x2: np.reshape(X2_test[k,:],[ 1,X2_test.shape[1] ]), keep_prob: 1 } )
                curr_test_label[ k,:]= np.round(curr_test_label[ k,:],decimals=3)
                print(k,"-th label probability=", curr_test_label[ k,:])
            
    print("\nOptimization Finished!")
    

# Calculate accuracy for test images
    print("Final Test Accuracy:", sess.run(accuracy, feed_dict={x1: X1_test[:,:,:,:], x2: X2_test[:,:], y: Y_test[:,:], keep_prob: 1.}))
    pred_test_label= y_conv.eval( session=sess, feed_dict={x1: X1_test[:,:,:,:], x2: X2_test[:,:], keep_prob: 1.} )
    
 
end_time= time.time()  
print("program time=", end_time-init_time) 


pickle.dump( train_acc_epoch, open( "train_acc_100epoch_3Lcnn_mfcc.p", "wb" ), protocol= 2 )
pickle.dump( test_acc_epoch, open( "test_acc_100epoch_3Lcnn_mfcc.p", "wb" ), protocol= 2 )
pickle.dump( test_pre_epoch, open( "test_pre_100epoch_3Lcnn_mfcc.p", "wb" ), protocol= 2 )
pickle.dump( test_rec_epoch, open( "test_rec_100epoch_3Lcnn_mfcc.p", "wb" ), protocol= 2 )
pickle.dump( test_f1s_epoch, open( "test_f1s_100epoch_3Lcnn_mfcc.p", "wb" ), protocol= 2 )

