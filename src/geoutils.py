

"""Module with useful functions to convert mesh to tfrecord."""

import numpy as np
import os
import math
import sys
import glob
import random
import trimesh

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

BOX_ID       = 0
CYLINDER_ID  = 1
TORUS_ID     = 2
CONE_ID      = 3
OTHER_ID     = 4
PRISM_ID     = 5


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def write_to_record(examples, filename):
  """
  Write an array of examples to single tfrecord
  """
  with tf.python_io.TFRecordWriter(filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

def color_to_id(color):
    if   color[0] == 255 and color[1] == 0 and color[2] == 0:
        return CYLINDER_ID
    elif color[0] == 0 and color[1] == 255 and color[2] == 0:
        return TORUS_ID
    elif color[0] == 255 and color[1] == 255 and color[2] == 0:
        return CONE_ID
    elif color[0] == 0 and color[1] == 0 and color[2] == 255:
        return BOX_ID
    elif color[0] == 255 and color[1] == 255 and color[2] == 255:
        return PRISM_ID

    return OTHER_ID

def mesh_to_example(mesh, input_labels=None):
  """
  Convert a mesh to example
  """

  graph = tf.Graph()
  
  with graph.as_default():
    
    vertices = tf.constant(mesh.vertices, dtype=tf.float32)

    if input_labels is not None: 
      labels = tf.constant(input_labels, dtype=tf.int32)
    else:
      labels = tf.constant(np.full(len(mesh.vertices), 2, dtype=np.int32), dtype=tf.int32)
    faces = tf.constant(mesh.faces, dtype=tf.int32)

    with tf.Session() as sess:
      str_vert = sess.run(tf.io.serialize_tensor(vertices))
      str_faces = sess.run(tf.io.serialize_tensor(faces))
      str_labels = sess.run(tf.io.serialize_tensor(labels))

  example = tf.train.Example(features=tf.train.Features(feature={
      'num_vertices': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(mesh.vertices)])),
      'num_triangles': tf.train.Feature(int64_list=tf.train.Int64List( value=[len(mesh.faces) ])),
      'vertices': tf.train.Feature( bytes_list=tf.train.BytesList( value=[str_vert])),
      'triangles': tf.train.Feature( bytes_list=tf.train.BytesList( value=[str_faces])),
      'labels': tf.train.Feature( bytes_list=tf.train.BytesList( value=[str_labels])),
    }))
  return example

def label_mesh(mesh):
  labels = np.zeros(len(mesh.vertices),dtype=np.int32)

  for i, color in enumerate(mesh.visual.vertex_colors):
    labels[i] = color_to_id(color)


  example = mesh_to_example(mesh, labels)
  return example

def label_meshes(scene):

  meshes = []
  labels_array = []

  idx_offset = 0

  vertices = None
  faces = None
  colors = None
  for g in scene.geometry.values():
    gcolors = np.full((len(g.vertices), 4), g.visual.material.main_color, dtype=np.int32)

    if vertices is None:
      vertices = g.vertices
    else:
      vertices = np.concatenate((vertices, g.vertices))
    
    if faces is None:
      faces = g.faces  
    else:
      for f in g.faces:
        f[0] += idx_offset
        f[1] += idx_offset
        f[2] += idx_offset
      faces = np.concatenate((faces, g.faces))
    idx_offset += len(g.vertices)

    if colors is None:
      colors = gcolors
    else:
      colors = np.concatenate((colors, gcolors))


  mesh = trimesh.Trimesh(vertices=vertices, 
                         faces=faces, 
                         vertex_colors=colors, 
                         process=False,
                         validate=False)
  
  example = label_mesh(mesh)
  return example, mesh

def convert_meshes_to_tfrecord(meshes, output_path):
  g = tf.Graph()

  with g.as_default():

    examples = []

    for mesh in meshes:
      example = label_mesh(mesh)
      examples.append(example)
    
    write_to_record(examples, output_path)
    
    ## Write to tfrecord
    write_to_record(examples, output_path)

