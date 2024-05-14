import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import gzip

F=np.load(open('stdp_conv_features_64-32_1.npz','rb'))
X=F['features_X']
mnist_file=np.load('mnist.npz')
(train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
y=train_labels

# F=pickle.load(gzip.open('../Spiking-CNN/EndToEnd_STDP_Spiking_CNN/RL8_Maps_1632.pickle.gz','rb'))
# X=np.array([x[1] for x in F['Maps']])
# X=X.reshape((X.shape[0], -1))
# y=F['Labels'][:X.shape[0]]

y=keras.utils.to_categorical(y, 10)
model=keras.Sequential([
    keras.Input(shape=(X.shape[1],)), 
    # keras.layers.Dense(1500, activation='relu'), 
    # keras.layers.Dropout(0.5), 
    # keras.layers.Dense(1500, activation='relu'), 
    # keras.layers.Dropout(0.5), 
    keras.layers.Dense(10,activation='softmax')
])
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
model.fit(X,y,batch_size=512, epochs=100, validation_split=0.2)
