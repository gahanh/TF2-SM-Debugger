import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
#import tensorflow as tf

import argparse
import os
import numpy as np
import json

import smdebug.tensorflow as smd

tf.compat.v1.disable_eager_execution()

def model():
    
    hook = smd.KerasHook.create_from_json_file()
    
       
    optimizer=tf.keras.optimizers.Adam()
    
    #opt = hook.wrap_optimizer(optimizer)
    
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(32, activation='relu', input_dim=32))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add an output layer with 10 output units:
    model.add(layers.Dense(10))
    
       
    model.compile(optimizer ,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    model.fit(data, labels, epochs=10, batch_size=32,callbacks=[hook])
    
    #model.fit(data, labels, epochs=10, batch_size=32)

    # With a Dataset
    #dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    #dataset = dataset.batch(32)

    #model.evaluate(dataset)
    
    #model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')

    return model

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument("--out_dir", type=str, default="./output")

    return parser.parse_known_args()



if __name__ == "__main__":
    args, unknown = _parse_args()  
    

    model()

    
    

