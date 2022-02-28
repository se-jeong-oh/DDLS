import json
import os
import sys
import tensorflow as tf
import numpy as np
import datetime

tf_config = {"cluster": {"worker": ["163.239.22.28:12345", "163.239.27.109:12345"]}, "task": {"index": 0, "type": "worker"}}
os.environ['TF_CONFIG']=json.dumps(tf_config)
#os.environ['GRPC_TRACE']='all'
os.environ['GRPC_VERBOSITY']='DEBUG'
#os.environ['GRPC_GO_LOG_SEVERITY_LEVEL']='info'
#os.environ['GRPC_GO_LOG_VERBOSITY_LEVEL']='2'
#os.environ['CGO_ENABLED']='1'
#os.environ['GRPC_TRACE']='http,api,client_channel_routing'
#os.environ['NCCL_DEBUG']='INFO'

tf.debugging.set_log_device_placement(True)
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver, communication_options=communication_options)

# every worker has different index, and same worker list
# tf_config = {
#    'cluster': {
#        'worker': ['163.239.27.109:50000', '163.239.22.28:50000']
#    },
#    'task': {'type': 'worker', 'index': 0}
#}
#os.environ['TF_CONFIG'] = json.dumps(tf_config)
#cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

# single worker test
# single_worker_dataset = mnist_dataset(batch_size)
# single_worker_model = build_and_compile_cnn_model()
# single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

# multi worker test
per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

#multi_worker_dataset = strategy.distribute_datasets_from_function(multi_worker_dataset)

multi_worker_model.fit(multi_worker_dataset, epochs=100, steps_per_epoch=70)


