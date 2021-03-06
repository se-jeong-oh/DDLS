import json
import os
import tensorflow as tf
import numpy as np
import datetime
import math
import time
from tensorflow.keras.utils import Sequence
from keras.layers import *
from keras.models import *
from keras.utils import *

# every worker has different index, and same worker list
tf_config = {
    'cluster': {
        'worker': ['163.239.27.109:50000', '163.239.22.28:50000']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['GRPC_TRACE']='all'
#os.environ['GRPC_VERBOSITY']='DEBUG'
#os.environ['GRPC_GO_LOG_SEVERITY_LEVEL']='info'
#os.environ['GRPC_GO_LOG_VERBOSITY_LEVEL']='2'
#os.environ['CGO_ENABLED']='1'
#os.environ['GRPC_TRACE']='http,api,client_channel_routing'
#os.environ['NCCL_DEBUG']='INFO'
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '500,520')

#cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
def cifar10_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  #train_dataset = tf.data.Dataset.from_tensor_slices(
  #    (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return (x_train, y_train)

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        start_time = time.time()
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        print("\nData Input Time: (us)", (time.time()-start_time)*1000000)
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self):
      self.indices = np.arange(len(self.x))
      if self.shuffle == True:
        np.random.shuffle(self.indices)

def build_and_compile_cnn_model():
  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=4, padding='same', strides=1, activation='relu', input_shape=(32,32,3)))
  model.add(MaxPool2D(pool_size=2))
  model.add(Conv2D(filters=32, kernel_size=4, padding='same', strides=1, activation='relu'))
  model.add(MaxPool2D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

# single worker test
# single_worker_dataset = mnist_dataset(batch_size)
# single_worker_model = build_and_compile_cnn_model()
# single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

# multi worker test
per_worker_batch_size = 64
#tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset_x, multi_worker_dataset_y = cifar10_dataset(global_batch_size)
train_loader = CIFAR10Sequence(multi_worker_dataset_x,multi_worker_dataset_y,global_batch_size,shuffle=True)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

#multi_worker_dataset = strategy.distribute_datasets_from_function(multi_worker_dataset)

multi_worker_model.fit(train_loader, epochs=10, verbose=0, callbacks=[tensorboard_callback])
