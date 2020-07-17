"""This module convert a mesh (in a format that is supported by trimesh) to a tfrecord."""

import os, sys

import trimesh
import numpy as np

from geoutils import convert_meshes_to_tfrecord, split_mesh



if len(sys.argv) <= 1:
    print("Usage: python mesh2tfrecord.py <file.tfrecord>")
    exit()
input_path = sys.argv[1]

MAX_SIZE = 1.7
mesh = trimesh.load(input_path, process=True)

meshes = mesh.split()
basename, _ = os.path.splitext(input_path)

out_path = basename + ".tfrecord"
convert_meshes_to_tfrecord(meshes, out_path)
