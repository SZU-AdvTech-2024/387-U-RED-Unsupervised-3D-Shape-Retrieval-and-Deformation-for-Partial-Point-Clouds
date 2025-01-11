#!/usr/bin/python

#------------------------------------------------------------------------------
# Define all the global variables for the project
#------------------------------------------------------------------------------

from __future__ import division
import os, sys
BASE_DIR = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__))))


g_renderer = '/mnt/d/20240308/U-RED/libigl-renderer/build/OSMesaRenderer'
g_azimuth_deg = -70
g_elevation_deg = 20
g_theta_deg = 0

g_partnet_dir = '/mnt/d/Dataset/PartNet/data_v0'
g_structurenet_root_dir = '/orion/u/mhsung/projects/deformation-space-learning/StructureNet'
g_structurenet_input_dir = "/mnt/d/20240308/U-RED/data/20240617/structure"

output_fol = "/orion/u/mikacuy/part_deform/"

g_structurenet_output_dir = "/mnt/d/20240308/U-RED/workspace/0812/data_aabb_all_models" #"/mnt/d/20240308/U-RED/data/20240617/data_aabb_all_models"
g_structurenet_output_dir_all = "/mnt/d/20240308/U-RED/workspace/0812/data_all_models" #"/mnt/d/20240308/U-RED/data/20240617/data_all_models"

# For pair generation
datasplits_file = "/mnt/d/20240308/U-RED/data/20240617/generated_datasplits/chair_-1.pickle"
all_part_path = "/mnt/d/20240308/U-RED/data/20240617/data_aabb_all_models/chair/h5"
src_obj_path = "/mnt/d/20240308/U-RED/data/20240617/data_aabb_all_models/chair/mesh"
tgt_path = "/mnt/d/20240308/U-RED/data/20240617/contrast_pair"
g_zero_tol = 1.0e-6
g_min_num_parts = 4
g_max_num_parts = 16
g_num_sample_points = 2048

file_src_connect = "/mnt/d/20240308/U-RED/workspace/0718/pickle/sources/sources_connect.npy"
sem_map = {"chair_base":"0", "chair_seat":"1", "chair_back":"2", "chair_arm":"3", "footrest":"9", "chair_head":"9"}
# g_num_sample_points = 8192

# For connectivity
g_adjacency_tol = 5.0e-2

label_to_idx = {'back_surface_vertical_bar': 0, 'arm_near_vertical_bar': 1, 'back_connector': 2, 'back_support': 3, 'arm_holistic_frame': 4, 'back_holistic_frame': 5, 'back_frame': 6, 'back_single_surface': 7, 'seat_surface_bar': 8, 'chair_base': 9, 'leg': 10, 'seat_frame_bar': 11, 'head_connector': 12, 'chair_arm': 13, 'bar_stretcher': 14, 'seat_surface': 15, 'seat_holistic_frame': 16, 'chair_head': 17, 'arm_sofa_style': 18, 'seat_single_surface': 19, 'regular_leg_base': 20, 'lever': 21, 'back_frame_vertical_bar': 22, 'arm_horizontal_bar': 23, 'arm_connector': 24, 'rocker': 25, 'foot': 26, 'back_surface': 27, 'arm_writing_table': 28, 'wheel': 29, 'caster_stem': 30, 'back_surface_horizontal_bar': 31, 'central_support': 32, 'back_frame_horizontal_bar': 33, 'seat_support': 34, 'star_leg_set': 35, 'seat_frame': 36, 'runner': 37, 'headrest': 38, 'pedestal': 39, 'footrest': 40, 'foot_base': 41}
