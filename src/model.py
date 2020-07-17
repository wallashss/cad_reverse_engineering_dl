
from tensorflow_graphics.nn.layer import graph_convolution as graph_conv
from tensorflow_graphics.notebooks import mesh_segmentation_dataio as dataio

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

MODEL_PARAMS = {
    'num_filters': 8,
    'num_classes': 4,
    'encoder_filter_dims': [32, 64, 128],
}

def mesh_encoder(batch_mesh_data, num_filters, output_dim, conv_layer_dims):

    """A mesh encoder using feature steered graph convolutions.

        The shorthands used below are
        `B`: Batch size.
        `V`: The maximum number of vertices over all meshes in the batch.
        `D`: The number of dimensions of input vertex features, D=3 if vertex
            positions are used as features.

    Args:
        batch_mesh_data: A mesh_data dict with following keys
        'vertices': A [B, V, D] `float32` tensor of vertex features, possibly
            0-padded.
        'neighbors': A [B, V, V] `float32` sparse tensor of edge weights.
        'num_vertices': A [B] `int32` tensor of number of vertices per mesh.
        num_filters: The number of weight matrices to be used in feature steered
        graph conv.
        output_dim: A dimension of output per vertex features.
        conv_layer_dims: A list of dimensions used in graph convolution layers.

    Returns:
        vertex_features: A [B, V, output_dim] `float32` tensor of per vertex
        features.
    """
    batch_vertices = batch_mesh_data['vertices']
  
    # Linear: N x D --> N x 16.
    vertex_features = tf.keras.layers.Conv1D(16, 1, name='lin16')(batch_vertices)

    # graph convolution layers
    for dim in conv_layer_dims:
        with tf.variable_scope('conv_%d' % dim):
            vertex_features = graph_conv.feature_steered_convolution_layer(
                vertex_features,
                batch_mesh_data['neighbors'],
                batch_mesh_data['num_vertices'],
                num_weight_matrices=num_filters,
                num_output_channels=dim,
                translation_invariant=True)
            vertex_features = tf.nn.relu(vertex_features)

    # Linear: N x 128 --> N x 256.
    vertex_features = tf.keras.layers.Conv1D(256, 1, name='lin256')(vertex_features)
    vertex_features = tf.nn.relu(vertex_features)

    # Linear: N x 256 --> N x output_dim.
    vertex_features = tf.keras.layers.Conv1D(output_dim, 1, name='lin_output')(vertex_features)

    return vertex_features

def get_learning_rate(params):
    """Returns a decaying learning rate."""
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        params['init_learning_rate'],
        global_step,
        params['lr_decay_steps'],
        params['lr_decay_rate'])
    return learning_rate

def model_fn(features, labels, mode, params):

    """Returns a mesh segmentation model_fn for use with tf.Estimator."""
    logits = mesh_encoder(features, params['num_filters'], params['num_classes'],
                            params['encoder_filter_dims'])
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    outputs = {
        'vertices': features['vertices'],
        'triangles': features['triangles'],
        'num_vertices': features['num_vertices'],
        'num_triangles': features['num_triangles'],
        'original_vertices': features['original_vertices'] if 'original_vertices' in features else [],
        'predictions': predictions,
    }
    # For predictions, return the outputs.
    if mode == tf.estimator.ModeKeys.PREDICT:
        outputs['labels'] = features['labels']
        return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)
    # Loss
    # Weight the losses by masking out padded vertices/labels.
    vertex_ragged_sizes = features['num_vertices']
    mask = tf.sequence_mask(vertex_ragged_sizes, tf.shape(labels)[-1])
    loss_weights = tf.cast(mask, dtype=tf.float32)

    loss_weights = tf.debugging.check_numerics(loss_weights, 'loss_weights')

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels, weights=loss_weights)


    # For training, build the optimizer.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer( learning_rate=get_learning_rate(params),
                                            beta1=params['beta'],
                                            epsilon=params['adam_epsilon'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # For eval, return eval metrics.
    eval_ops = {
        'mean_loss': tf.metrics.mean(loss),
        'accuracy':tf.metrics.accuracy(labels=labels, predictions=predictions, weights=loss_weights)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_ops)