
"""This module convert a mesh (in a format that is supported by trimesh) to a tfrecord."""

import glob
import os
import sys
import trimesh
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow as tf2
from tensorflow_graphics.notebooks import mesh_segmentation_dataio as dataio
from tensorflow_graphics.geometry.convolution.utils import flatten_batch_to_2d

from model import mesh_encoder, MODEL_PARAMS, model_fn



tf.disable_v2_behavior()


local_model_dir = "model/reverse_eng_model"

test_io_params = {
    'is_training': False,
    'sloppy': False,
    'shuffle': False,
    'repeat': False,
    'batch_size': 8,
    'mean_center': False
}

# This scale we found experimentally
# It is a "safe" scale for triangles
# to perform graph convolution on them.
# This is a possible bug in tensorgraphics.
MAX_SIZE = 1.7


# Get the input
if len(sys.argv) <= 1:
    print("Usage: python test.py <file.tfrecord>")
    exit()

test_tfrecords = sys.argv[1]


# 0 - Box - blue
# 1 - Cylinder - red
# 2 - Torues - green
# 3 - Cone - yellow
# 4 - Other - Purple
color_map = np.array([[0, 0, 1], 
                      [1, 0, 0], 
                      [0, 1, 0], 
                      [1, 1, 0], 
                      [1, 0, 1], 
                      [1, 0, 1]], dtype=np.float32)

def input_fn():

    mesh_data, _ = dataio.create_input_from_dataset( dataio.create_dataset_from_tfrecords, test_tfrecords, test_io_params)

    vertices = mesh_data['vertices']

    # Save the origin vertices to ease the reconstruction
    mesh_data['original_vertices'] = vertices


    vertices_sizes = tf2.cast(mesh_data['num_vertices'], tf2.float32)
    batch_size = tf.shape(vertices)[0]

    # Calculate mesh center
    vertices_sizes = tf.reshape(vertices_sizes, [ batch_size, 1, 1])

    centers = tf.reduce_sum(vertices, axis=[1], keepdims=True)
    centers = centers / vertices_sizes

    # Center the mesh to origin
    vertices -= centers

    # This trick fix the centering 
    # The transformation to center will create
    # messy vertices 
    vertices, unflatten = flatten_batch_to_2d(vertices, mesh_data['num_vertices'])
    vertices = unflatten(vertices)
    
    # Calculate the mesh size
    max = tf.reduce_max(vertices, axis=[1], keepdims=True)
    min = tf.reduce_min(vertices, axis=[1], keepdims=True)
    extents = max - min

    max_dim = tf.reduce_max(extents, axis=[2], keepdims=True)

    scale = MAX_SIZE / max_dim 
    
    # Scale the mesh
    vertices *= scale


    # Update vertices
    mesh_data['vertices'] = vertices

    return mesh_data


# Load the model
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=local_model_dir,
                                   params=MODEL_PARAMS)

# Execute the prediction
test_predictions = estimator.predict(input_fn)

# Rebuild the mesh
meshes = []
for i, prediction in enumerate(test_predictions):
    
    vertices = prediction['vertices']
    triangles = prediction['triangles']
    num_vertices = prediction['num_vertices']
    num_triangles = prediction['num_triangles']

    vertices = prediction['original_vertices']

    predicted_mesh_data = trimesh.Trimesh(vertices=vertices,
                                        faces=prediction['triangles'],
                                        vertex_colors=color_map[prediction['predictions']], 
                                        process=False)
    meshes.append(predicted_mesh_data)

mesh = trimesh.util.concatenate(meshes)

# Scale the mesh if it is too big
max_dim = np.max(mesh.extents)
if max_dim > 1e3:
    mesh.apply_transform(trimesh.transformations.scale_matrix( 1 / max_dim))

# Render the mesh
mesh.show()