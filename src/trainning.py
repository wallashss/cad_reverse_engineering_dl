# Import everything

import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


import numpy as np
import trimesh
import shutil

from tensorflow_graphics.notebooks import mesh_segmentation_dataio as dataio


from model import mesh_encoder, MODEL_PARAMS, model_fn


BASE_DIR = "dataset"

train_data_files = glob.glob(os.path.join(BASE_DIR, "_datasets/train_*.tfrecord"))
test_data_files  = glob.glob(os.path.join(BASE_DIR, "_datasets/tests_*.tfrecord"))


# Model destination
retrain_model_dir = 'model/reverse_eng_model'

os.makedirs(retrain_model_dir, exist_ok=True)


# Model Parameters
train_io_params = {
    'batch_size': 16,
    'parallel_threads': 16,
    'is_training': True,
    'shuffle': True,
    'sloppy': True,
}

eval_io_params = {
    'batch_size': 16,
    'parallel_threads': 16,
    'is_training': False,
    'shuffle': True
}


def train_fn():
  return dataio.create_input_from_dataset(dataio.create_dataset_from_tfrecords,
                                          train_data_files, train_io_params)


def eval_fn():
  return dataio.create_input_from_dataset(dataio.create_dataset_from_tfrecords,
                                          test_data_files, eval_io_params)


train_params = {
    'beta': 0.9,
    'adam_epsilon': 1e-8,
    'init_learning_rate': 1e-3,
    'lr_decay_steps': 10000,
    'lr_decay_rate': 0.95,
}

train_params.update(MODEL_PARAMS)

checkpoint_delay = 120  # Checkpoint every 2 minutes.
max_steps = 100000  # Number of training steps.

config = tf.estimator.RunConfig(log_step_count_steps=1,
                                save_checkpoints_secs=checkpoint_delay,
                                keep_checkpoint_max=3)

classifier = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir=retrain_model_dir,
                                    config=config,
                                    params=train_params)

train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=max_steps)

eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn,
                                  steps=None,
                                  start_delay_secs=2 * checkpoint_delay,
                                  throttle_secs=checkpoint_delay)

print('Start training & eval.')
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
print('Train and eval done.')