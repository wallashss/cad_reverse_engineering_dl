
import numpy as np
import os
import math
import sys
import glob
import random
import trimesh

from geoutils import label_mesh, convert_meshes_to_tfrecord, write_to_record

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

BASE_DIR = os.path.normpath('dataset')

print(BASE_DIR)

MAX_SIZE = 1.7


os.makedirs(os.path.join(BASE_DIR, "_datasets"), exist_ok=True)


label_path = os.path.join(BASE_DIR, "_labels") 

boxes     = glob.glob(os.path.join(label_path, "boxes/*.*"))
cones     = glob.glob(os.path.join(label_path, "cones/*.*"))
cylinders = glob.glob(os.path.join(label_path, "cylinders/*.*"))
pmeshes    = glob.glob(os.path.join(label_path, "meshes/*.*"))
toruses   = glob.glob(os.path.join(label_path, "torus/*.*"))


box_color = [0, 0, 255, 255]
cone_color = [255, 255, 0, 255]
cyl_color = [255, 0, 0, 255]
mesh_color = [255, 0, 255, 255]
torus_color = [0, 255, 0, 255]


# Method to generate dataset
def draw_geometry(collection, color, count):
    
    meshes = []

    for _ in range(count):
        path = random.choice(collection)
        mesh = trimesh.load_mesh(path)
        mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), color)
        center = mesh.centroid

        # Center geometry to origin
        matrix = trimesh.transformations.translation_matrix(center)
        matrix_inv = trimesh.transformations.inverse_matrix(matrix)
        mesh.apply_transform(matrix_inv)

        max_size = np.max(mesh.extents)

        scale = MAX_SIZE / max_size
        
        # Reduce scale
        mesh.apply_transform(trimesh.transformations.scale_matrix(scale))

        # Apply random rotation
        mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

        # TODO Not working right now
        # Apply random translation
        # random_translation = (np.random.random((1, 3)) - np.array([0.5, 0.5, 0.5])) * 100
        # mesh.apply_transform(trimesh.transformations.translation_matrix(random_translation[0]))

        meshes.append(mesh)

    return trimesh.util.concatenate(meshes)


def generate_dataset(size):

    meshes = []
    for i in range(size):
        if i > 0 and i % 10 == 0:
          print("Step {}/{}".format(i, size))
        a = draw_geometry(boxes, box_color, 6)
        b = draw_geometry(cylinders, cyl_color, 3)
        c = draw_geometry(cones, cone_color, 6)
        d = draw_geometry(toruses, torus_color, 1)
        mesh = trimesh.util.concatenate([a, b, c, d])

        meshes.append(mesh)
    
    print("Step {}/{}".format(size, size))
    return meshes


TRAIN_DATASET_SIZE = 10000
TESTS_DATASET_SIZE = 500

train_path = os.path.join(BASE_DIR, "_datasets/train_{}.tfrecord".format(TRAIN_DATASET_SIZE)) 
tests_path = os.path.join(BASE_DIR, "_datasets/tests_{}.tfrecord".format(TESTS_DATASET_SIZE)) 



print("Generating training")
train_meshes = generate_dataset(TRAIN_DATASET_SIZE)
print("Writing tfrecord", train_path)
convert_meshes_to_tfrecord(train_meshes, train_path)

print("Generating testing")
train_meshes = generate_dataset(TESTS_DATASET_SIZE)
print("Writing tfrecord", train_path)
convert_meshes_to_tfrecord(train_meshes, tests_path)