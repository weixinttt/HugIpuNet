

from tensorflow.keras.layers import Permute, Reshape, Input, Lambda
import time
import keras.callbacks as Callback
import h5py
import math
import pylab
import keras
import numpy as np
import scipy.io as sio
import scipy.io as scio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score
from mamba import Mamba,ResidualBlock,MambaConfig,MambaBlock,RMSNorm
from mamba1 import Mamba1,ResidualBlock1,MambaConfig1,MambaBlock1,RMSNorm1
from keras.layers import Activation,Input,Dense,Lambda,Conv2D,Concatenate,Permute,concatenate,AveragePooling2D,MaxPooling2D,BatchNormalization,Reshape,Multiply,Add,Conv3D,Flatten,Dropout,GlobalAveragePooling2D
iterations=20000
from spektral.layers import GCNConv,GATConv,ChebConv
from skimage.segmentation import slic
from scipy.ndimage import mean
from keras import Model



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.config.experimental_run_functions_eagerly(True)

dataset=0

if dataset==0:
    data=h5py.File('MUUFL_plus.mat', 'r')
    batchsize=128
    source='MUUFLresult2/'
if dataset==1:
    data=h5py.File('Italy_plus.mat', 'r')
    source='result2/'
    batchsize=256

X_spatial_all=data['X_spatial'][()].transpose(3,2,1,0)
LiDAR_all=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all=np.expand_dims(LiDAR_all,-1)
act_Y_train_all=data['act_Y_train'][()].transpose(1,0)
indexi_all=data['indexi'][()].transpose(1,0)
indexj_all=data['indexj'][()].transpose(1,0)

X_spatial_all=X_spatial_all.astype('float32')
LiDAR_all=LiDAR_all.astype('float32')
act_Y_train_all=act_Y_train_all.astype('int')
indexi_all=indexi_all.astype('float32')
indexj_all=indexj_all.astype('float32')





act_Y_train_all[act_Y_train_all==-1]=0

slide_size=5
if slide_size==3:
    X_spatial_all=X_spatial_all[:,4:7,4:7,:]
    LiDAR_all=LiDAR_all[:,4:7,4:7,:]
if slide_size==5:
    X_spatial_all=X_spatial_all[:,3:8,3:8,:]
    LiDAR_all=LiDAR_all[:,3:8,3:8,:]
if slide_size==7:
    X_spatial_all=X_spatial_all[:,2:9,2:9,:]
    LiDAR_all=LiDAR_all[:,2:9,2:9,:]
if slide_size==9:
    X_spatial_all=X_spatial_all[:,1:10,1:10,:]
    LiDAR_all=LiDAR_all[:,1:10,1:10,:]
if slide_size==11:
    X_spatial_all=X_spatial_all[:,:,:,:]
    LiDAR_all=LiDAR_all[:,:,:,:]
###############################################################################







scaler=MinMaxScaler(feature_range=(0,1))

X_spatial_all_=X_spatial_all.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1]*X_spatial_all.shape[2]*X_spatial_all.shape[3]])
LiDAR_all_=LiDAR_all.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1]*LiDAR_all.shape[2]*LiDAR_all.shape[3]])

X_spatial_all_=scaler.fit_transform(X_spatial_all_)
LiDAR_all_=scaler.fit_transform(LiDAR_all_)

X_spatial_all=X_spatial_all_.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1],X_spatial_all.shape[2],X_spatial_all.shape[3]])
LiDAR_all=LiDAR_all_.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1],LiDAR_all.shape[2],LiDAR_all.shape[3]])
#####################     选择训练集      ######################################
#随机选取
act_Y_train_all=np.reshape(act_Y_train_all,act_Y_train_all.shape[0])
indexi_all=np.reshape(indexi_all,indexi_all.shape[0])
indexj_all=np.reshape(indexj_all,indexj_all.shape[0])

act_Y_train=act_Y_train_all

randpaixv=act_Y_train_all.argsort()
X_spatial_all=X_spatial_all[randpaixv]
LiDAR_all=LiDAR_all[randpaixv]
indexi_all=indexi_all[randpaixv]
indexj_all=indexj_all[randpaixv]
act_Y_train_all=act_Y_train_all[randpaixv]

X_spatial_all=X_spatial_all[act_Y_train_all>0]
LiDAR_all=LiDAR_all[act_Y_train_all>0]
indexi_all=indexi_all[act_Y_train_all>0]
indexj_all=indexj_all[act_Y_train_all>0]
act_Y_train_all=act_Y_train_all[act_Y_train_all>0]

indices=np.arange(X_spatial_all.shape[0])
indices_train,indices_test,act_Y_train_train,act_Y_train_test=train_test_split(indices,act_Y_train_all,test_size=0.99,stratify=act_Y_train_all)

X_spatial_train=X_spatial_all[indices_train,:,:]
LiDAR_train=LiDAR_all[indices_train,:,:]
act_Y_train_train=act_Y_train_all[indices_train]
indexi_train=indexi_all[indices_train]
indexj_train=indexj_all[indices_train]

X_spatial_test=X_spatial_all[indices_test,:,:]
LiDAR_test=LiDAR_all[indices_test,:,:]
act_Y_train_test=act_Y_train_all[indices_test]
indexi_test=indexi_all[indices_test]
indexj_test=indexj_all[indices_test]

act_Y_train_train=np_utils.to_categorical(act_Y_train_train-1)
act_Y_train_test=np_utils.to_categorical(act_Y_train_test-1)




#################################
def sampling(args):
    z_mean,z_log_var=args
    epsilon=K.random_normal(shape=K.shape(z_mean))
    return z_mean+K.exp(z_log_var/2)*epsilon
class EpochCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()
        
    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.sum_time = sum(self.times)





#
def dynamic_image_to_weighted_graph(node_features, width, sigma=1.0,beta=0.1, time_step=0,external_info=None,threshold=0.3):
    
    batch_size = tf.shape(node_features)[0]
    num_nodes = width * width

    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    all_edges = []

   
    for i in range(width):
        for j in range(width):
            current_node = i * width + j
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < width and 0 <= nj < width:
                    neighbor_node = ni * width + nj
                    all_edges.append((current_node, neighbor_node))

    edges = tf.constant(all_edges, dtype=tf.int32)

    source_nodes = edges[:, 0]
    target_nodes = edges[:, 1]
    
    if external_info is not None:
       
        diff = node_features - external_info  
        euclidean_distance = tf.norm(diff, axis=-1)  

        active_nodes = tf.cast(euclidean_distance < threshold, tf.float32)  
        

        updated_node_features = tf.where(tf.expand_dims(active_nodes, -1) > 0, external_info, node_features)
    else:
        updated_node_features = node_features


    source_features = tf.gather(updated_node_features, source_nodes, axis=1)
    target_features = tf.gather(updated_node_features, target_nodes, axis=1)

    distances = tf.norm(source_features - target_features, axis=-1)
    
    

    dynamic_sigma = sigma + tf.sin(beta * tf.cast(time_step, tf.float32)) 
    weights = tf.exp(-tf.square(distances) / (2 * dynamic_sigma ** 2))

    batch_indices = tf.repeat(tf.range(batch_size), tf.shape(source_nodes)[0])
    row_indices = tf.tile(source_nodes, [batch_size])
    col_indices = tf.tile(target_nodes, [batch_size])
    indices = tf.stack([batch_indices, row_indices, col_indices], axis=1)

    weighted_adj_matrix = tf.scatter_nd(indices, tf.reshape(weights, [-1]), [batch_size, num_nodes, num_nodes])


    epsilon = 1e-8
    degree_matrix = tf.reduce_sum(weighted_adj_matrix, axis=-1, keepdims=True)
    degree_inv_sqrt = tf.math.pow(degree_matrix + epsilon, -0.5)

  
    degree_inv_sqrt = tf.where(tf.math.is_finite(degree_inv_sqrt), degree_inv_sqrt, tf.zeros_like(degree_inv_sqrt))

  
    normalized_adj_matrix = degree_inv_sqrt * weighted_adj_matrix * tf.transpose(degree_inv_sqrt, perm=[0, 2, 1])

    return normalized_adj_matrix







#################################
activation='tanh'
kernel_regularizer=tf.keras.regularizers.l2(0.01)
lr=0.0005
hidden = 16
widths = 5
epochs = 100

##############################################################################
#网络开始
H1_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))

H2_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))

L1_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))

L2_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))



indexi1=Input(shape=(1,))
indexi2=Input(shape=(1,))

indexj1=Input(shape=(1,))
indexj2=Input(shape=(1,))







mamba1 = Mamba(MambaConfig(d_model=X_spatial_train.shape[-1],n_layers=1))
mamba2 = Mamba1(MambaConfig1(d_model=slide_size*slide_size,n_layers=1))
gcn = GCNConv(channels=hidden,activation='relu')
gcn2 = GCNConv(X_spatial_train.shape[-1], activation='relu')

#########################################################################
H1_spa=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H1_input)
H1_spa=BatchNormalization()(H1_spa)
H1_spa=Activation(activation)(H1_spa)





H1_spe_conv=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H1_input)
H1_spe_conv=BatchNormalization()(H1_spe_conv)
H1_spe_conv=Activation(activation)(H1_spe_conv)


H1_spe_train = H1_spe_conv.shape


H1_spe_conv = K.reshape(H1_spe_conv, (-1, H1_spe_train[1] * H1_spe_train[2], H1_spe_train[3]))

H1_spe = K.reshape(H1_spe_conv, (-1, H1_spe_train[1] * H1_spe_train[2], H1_spe_train[3]))


for epoch in range(epochs):
    time_step = epoch 
    adj_H1_spe = dynamic_image_to_weighted_graph(H1_spe,width=widths,time_step=time_step)

H1_spe = gcn([H1_spe,adj_H1_spe])
H1_spe = gcn2([H1_spe,adj_H1_spe])

H1_spe=BatchNormalization()(H1_spe)
H1_spe=Activation(activation)(H1_spe)


H1_spe_mamba = K.permute_dimensions(H1_spe_conv, (0, 2, 1))
H1_spe_mamba = mamba2(H1_spe_mamba)
H1_spe_mamba=BatchNormalization()(H1_spe_mamba)
H1_spe_mamba=Activation(activation)(H1_spe_mamba)



H1_spe_mamba = K.permute_dimensions(H1_spe_mamba, (0, 2, 1))

H1_spe = Multiply()([H1_spe,H1_spe_mamba])










R1=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H1_spa)
R1=BatchNormalization()(R1)
R1=Activation(activation)(R1)




S1=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H1_spa)
S1=BatchNormalization()(S1)
S1=Activation(activation)(S1)






R1_train = R1.shape
R1_conv = K.reshape(R1, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], R1_train[3]))
R1 = K.reshape(R1, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], R1_train[3]))




S1_train = S1.shape
S1_conv = K.reshape(S1, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], S1_train[3]))
S1 = K.reshape(S1, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], S1_train[3]))














#########################################################################


H2_spa=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H2_input)
H2_spa=BatchNormalization()(H2_spa)
H2_spa=Activation(activation)(H2_spa)




H2_spe_conv=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H2_input)
H2_spe_conv=BatchNormalization()(H2_spe_conv)
H2_spe_conv=Activation(activation)(H2_spe_conv)


H2_spe_train = H2_spe_conv.shape


H2_spe_conv = K.reshape(H2_spe_conv, (-1, H2_spe_train[1] * H2_spe_train[2], H2_spe_train[3]))

H2_spe = K.reshape(H2_spe_conv, (-1, H2_spe_train[1] * H2_spe_train[2], H2_spe_train[3]))


for epoch in range(epochs):
    time_step = epoch 
    adj_H2_spe = dynamic_image_to_weighted_graph(H2_spe,width=widths,time_step=time_step)

H2_spe = gcn([H2_spe,adj_H2_spe])
H2_spe = gcn2([H2_spe,adj_H2_spe])

H2_spe=BatchNormalization()(H2_spe)
H2_spe=Activation(activation)(H2_spe)


H2_spe_mamba = K.permute_dimensions(H2_spe_conv, (0, 2, 1))
H2_spe_mamba = mamba2(H2_spe_mamba)
H2_spe_mamba=BatchNormalization()(H2_spe_mamba)
H2_spe_mamba=Activation(activation)(H2_spe_mamba)



H2_spe_mamba = K.permute_dimensions(H2_spe_mamba, (0, 2, 1))

H2_spe = Multiply()([H2_spe,H2_spe_mamba])




R2=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H2_spa)
R2=BatchNormalization()(R2)
R2=Activation(activation)(R2)


S2=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H2_spa)
S2=BatchNormalization()(S2)
S2=Activation(activation)(S2)



R2_train = R2.shape


R2_conv = K.reshape(R2, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], R2_train[3]))
R2 = K.reshape(R2, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], R2_train[3]))

S2_train = S2.shape


S2_conv = K.reshape(S2, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], S2_train[3]))
S2 = K.reshape(S2, (-1, X_spatial_train.shape[1] * X_spatial_train.shape[2], S2_train[3]))




#########################################################################

R1 = mamba1(R1)
R1=BatchNormalization()(R1)
R1=Activation(activation)(R1)

R1 = Multiply()([R1,R1_conv])


S1 = mamba1(S1)
S1=BatchNormalization()(S1)
S1=Activation(activation)(S1)



#########################################################################


R2 = mamba1(R2)
R2=BatchNormalization()(R2)
R2=Activation(activation)(R2)

R2 = Multiply()([R2,R2_conv])




S2 = mamba1(S2)
S2=BatchNormalization()(S2)
S2=Activation(activation)(S2)




###########################################################################

L1=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L1_input)
L1=BatchNormalization()(L1)
L1=Activation(activation)(L1)



L1_train = L1.shape

L1 = K.reshape(L1, (-1, L1_train[1] * L1_train[2], L1_train[3]))


L1_conv = K.reshape(L1, (-1, L1_train[1] * L1_train[2], L1_train[3]))






for epoch in range(epochs):
    time_step = epoch 
    adj_L1 = dynamic_image_to_weighted_graph(L1,width=widths,time_step=time_step,external_info=L1)

L1 = gcn([L1,adj_L1])
L1 = gcn2([L1,adj_L1])

L1=BatchNormalization()(L1)
L1=Activation(activation)(L1)


L1 = mamba1(L1)
L1=BatchNormalization()(L1)
L1=Activation(activation)(L1)



###########################################################################



L2=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L2_input)
L2=BatchNormalization()(L2)
L2=Activation(activation)(L2)


L2=Conv2D(X_spatial_all.shape[-1],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L2)
L2=BatchNormalization()(L2)
L2=Activation(activation)(L2)




L2_train = L2.shape

L2 = K.reshape(L2, (-1, L2_train[1] * L2_train[2], L2_train[3]))


L2_conv = K.reshape(L2, (-1, L2_train[1] * L2_train[2], L2_train[3]))




for epoch in range(epochs):
    time_step = epoch 
    adj_L2 = dynamic_image_to_weighted_graph(L2,width=widths,time_step=time_step)

L2 = gcn([L2,adj_L2])
L2 = gcn2([L2,adj_L2])

L2=BatchNormalization()(L2)
L2=Activation(activation)(L2)


L2 = mamba1(L2)
L2=BatchNormalization()(L2)
L2=Activation(activation)(L2)




R1_train = R1.shape

R1 = K.reshape(R1, (-1, X_spatial_train.shape[1] , X_spatial_train.shape[2], R1_train[2]))


R2_train = R2.shape

R2 = K.reshape(R2, (-1, X_spatial_train.shape[1] , X_spatial_train.shape[2], R2_train[2]))




S1_train = S1.shape

S1 = K.reshape(S1, (-1, X_spatial_train.shape[1] , X_spatial_train.shape[2], S1_train[2]))




S2_train = S2.shape

S2 = K.reshape(S2, (-1, X_spatial_train.shape[1] , X_spatial_train.shape[2], S2_train[2]))



H1_spe_train = H1_spe.shape

H1_spe = K.reshape(H1_spe, (-1, X_spatial_train.shape[1] , X_spatial_train.shape[2], H1_spe_train[2]))



H2_spe_train = H2_spe.shape

H2_spe = K.reshape(H2_spe, (-1, X_spatial_train.shape[1] , X_spatial_train.shape[2], H2_spe_train[2]))




L1_train = L1.shape

L1 = K.reshape(L1, (-1, LiDAR_train.shape[1] , LiDAR_train.shape[2], L1_train[2]))

L2_train = L2.shape

L2 = K.reshape(L2, (-1, LiDAR_train.shape[1] , LiDAR_train.shape[2], L2_train[2]))



R1_spe = Concatenate()([H1_spe,R1])
S_L = Concatenate()([S1,L1])


R_out = Concatenate()([R1_spe,S_L])

R_out = GlobalAveragePooling2D()(R_out)

R_out=Dense(act_Y_train_train.shape[1],activation='softmax')(R_out)


##############################################################################

Hx1_middle=GlobalAveragePooling2D()(H1_input)
Hx2_middle=GlobalAveragePooling2D()(H2_input)


Lx1_middle=GlobalAveragePooling2D()(L1_input)
Lx2_middle=GlobalAveragePooling2D()(L2_input)




R1_mean=GlobalAveragePooling2D()(R1)
R2_mean=GlobalAveragePooling2D()(R2)


S1_mean=GlobalAveragePooling2D()(S1)
S2_mean=GlobalAveragePooling2D()(S2)


Lx1_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train.shape[3])]))(Lx1_middle)
Lx2_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train.shape[3])]))(Lx2_middle)

HSI_gradient=Lambda(lambda x:(1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9))))([Hx1_middle,Hx2_middle,indexi1,indexi2,indexj1,indexj2])
LiDAR_gradient=Lambda(lambda x:(1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9))))([Lx1_middle,Lx2_middle,indexi1,indexi2,indexj1,indexj2])

HSI_gradient=Activation('sigmoid')(HSI_gradient)
LiDAR_gradient=Activation('sigmoid')(LiDAR_gradient)

mid1=Multiply()([R1_mean,S1_mean])
mid2=Multiply()([R2_mean,S2_mean])


loss1=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss1')([Hx1_middle,mid1])
loss2=Lambda(lambda x:K.mean(K.square(x[0]-x[1]),axis=-1),name='loss2')([Hx2_middle,mid2])

loss3=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss3')([R1_mean,R2_mean,HSI_gradient])
loss4=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss4')([S1_mean,S2_mean,LiDAR_gradient])

loss_RS=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([loss1,loss2,loss3,loss4])





network=keras.models.Model([H1_input,H2_input,L1_input,L2_input,indexi1,indexi2,indexj1,indexj2],[R_out,loss_RS])
network.compile(loss=['categorical_crossentropy','mean_squared_error'],
                loss_weights=[1,0.0001],
                optimizer=tf.keras.optimizers.Adam(lr=lr,decay=0.01)
                )
network.summary()
##############################################################
maxacc=0
maxkappa=0
maxp=0
maxr=0
maxf1_score=0
epoch_time_callback = EpochCallback()

for iter in range(iterations):
    index1=[ind for ind in range(int(X_spatial_train.shape[0]))]
    np.random.shuffle(index1)
    X_spatial_train1=X_spatial_train[index1]
    LiDAR_train1=LiDAR_train[index1]
    act_Y_train_train1=act_Y_train_train[index1]
    indexi_train1=indexi_train[index1]
    indexj_train1=indexj_train[index1]  
      
    index2=[ind for ind in range(int(X_spatial_train.shape[0]))]
    np.random.shuffle(index2)
    X_spatial_train2=X_spatial_train[index2]
    LiDAR_train2=LiDAR_train[index2]
    act_Y_train_train2=act_Y_train_train[index2]
    indexi_train2=indexi_train[index2]
    indexj_train2=indexj_train[index2]
    
    history=network.fit([X_spatial_train1,
                         X_spatial_train2,
                         LiDAR_train1,
                         LiDAR_train2,
                         indexi_train1,
                         indexi_train2,
                         indexj_train1,
                         indexj_train2],
                        [act_Y_train_train1,
                         np.zeros([act_Y_train_train.shape[0],1])],
                        batch_size=batchsize,
                        epochs=epochs,
                        shuffle=True,
                        verbose=1,
                        callbacks=[epoch_time_callback])
    Test_loss=network.predict([X_spatial_test,X_spatial_test,LiDAR_test,LiDAR_test,indexi_test,indexi_test,indexj_test,indexj_test])
    predicted=Test_loss[0]
    
    
    predicted_label=predicted.argmax(axis=1)
    raw_label=act_Y_train_test.argmax(axis=1)

    acc=accuracy_score(predicted_label,raw_label)
    
    if acc>0:
        acc=acc
        Pred_result_ = predicted_label
        generator_component=keras.models.Model([H1_input,H2_input,L1_input,L2_input,indexi1,indexi2,indexj1,indexj2],[R1,R2,S1,S2])
        [R1_result,R2_result,S1_result,S2_result]=generator_component.predict([X_spatial_all,X_spatial_all,LiDAR_all,LiDAR_all,indexi_all,indexi_all,indexj_all,indexj_all])
        R=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            R[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=R1_result[iii,int((R1_result.shape[1]-1)/2),int((R1_result.shape[1]-1)/2),:]
        S=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            S[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=S1_result[iii,int((S1_result.shape[1]-1)/2),int((S1_result.shape[1]-1)/2),:]            

        
        maxacc=acc
        maxOA=maxacc
        maxAllAcc=recall_score(raw_label,predicted_label,average=None)
        maxAA=recall_score(raw_label,predicted_label,average='macro')
        maxkappa=cohen_kappa_score(np.array(predicted_label).reshape(-1,1),np.array(raw_label).reshape(-1,1))
        maxp=precision_score(raw_label,predicted_label,average='macro')
        maxf1score=f1_score(raw_label,predicted_label,average='macro')
        

        MAP=np.zeros([int(indexi_all.max()),int(indexj_all.max())])
        for ii in range(act_Y_train_test.shape[0]):
            MAP[int(indexi_test[ii]-1),int(indexj_test[ii]-1)]=predicted.argmax(axis=1)[ii]+1
        for ii in range(act_Y_train_train.shape[0]):
            MAP[int(indexi_train[ii]-1),int(indexj_train[ii]-1)]=(act_Y_train_train.argmax(axis=1))[ii]+1
        

        name=source+'net_result_'+str(iter)+'_'+str(maxacc)+'.mat'
        sio.savemat(name, {'Pred_result': predicted_label,
                           'raw_label':raw_label,
                           'maxacc':maxacc,
                           'maxAllAcc':maxAllAcc,
                           'maxOA':maxOA,
                           'maxAA':maxAA,
                           'maxkappa':maxkappa,
                           'maxp':maxp,
                           'maxf1score':maxf1score,
                            'R':R,
                            'S':S,
                           'MAP':MAP,
                           'total_time':epoch_time_callback.sum_time
                           })
