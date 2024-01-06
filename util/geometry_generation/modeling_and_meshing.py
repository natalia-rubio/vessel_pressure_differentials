import os
import sys
import math
import numpy as np
import sv
import pickle
from sv import *
import vtk
import platform
from util.geometry_generation.meshing_util import *
import copy
import pdb

def construct_model(model_name, segmentations, geo_params):
    
    contour_list = segmentations
    capped_vessels = create_vessels(contour_list=contour_list)
    unioned_model = union_all(capped_vessels)
    model = clean(unioned_model)
    model = norm(model)
    #model.write("junction_model_normed", "vtp")
    # smooth_model = model.get_polydata()
    # smoothing_params = {'method':'constrained', 'num_iterations':30}
    # smooth_model = sv.geometry.local_sphere_smooth(surface = smooth_model, radius = 1, center = [0, 0, 0], smoothing_parameters = smoothing_params)
    # # [=== Combine faces ===]
    # #
    # model.set_surface(smooth_model)
    print("Surface set.")
    model.compute_boundary_faces(85)
    model, walls, caps, ids = combine_walls(model)
    model = combine_caps(model, walls, ids, num_caps = 2)
    print("boundary faces computed")
    #model.write("junction_model", "vtp")

    return model

def get_mesh(model_name, model, geo_params, anatomy, set_type, mesh_divs = 3):
    min_dimension = min([geo_params["inlet_radius"], geo_params["outlet_radius"]])
    edge_size = min_dimension/mesh_divs 
    caps = model.identify_caps()
    ids = model.get_face_ids()
    walls = [ids[i] for i,x in enumerate(caps) if not x]
    faces = model.get_face_ids()
    cap_faces = [ids[i] for i,x in enumerate(caps) if x]
    print(cap_faces)

    mesher = sv.meshing.create_mesher(sv.meshing.Kernel.TETGEN)
    mesher.set_model(model)
    mesher.set_boundary_layer_options(number_of_layers=2, edge_size_fraction=0.5, layer_decreasing_ratio=0.8, constant_thickness=False)
    options = sv.meshing.TetGenOptions(global_edge_size = edge_size, surface_mesh_flag=True, volume_mesh_flag=True)
    options.optimization = 10
    options.quality_ratio = 1
    options.no_bisect = True
    options.minimum_dihedral_angle = 18.0

    print("Options values: ")
    [ print("  {0:s}:{1:s}".format(key,str(value))) for (key, value) in sorted(options.get_values().items()) ]

    mesher.set_walls(walls)
    mesher.generate_mesh(options)

    msh = mesher.get_mesh()
    generate_mesh_complete_folder(model, mesher, model_name, caps, ids, walls, faces, anatomy, set_type)
    return msh, model

def generate_mesh_complete_folder(model, mesher, model_name, caps, ids, walls, faces, anatomy, set_type):
    print("Generating mesh complete folder. \n")
    home_dir = os.path.expanduser("~")
    dir = "data/synthetic_junctions/" + anatomy + "/" + set_type + "/" + model_name

    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir + '/mesh-complete'):
        os.mkdir(dir +  '/mesh-complete')
    if not os.path.exists(dir + '/mesh-complete/mesh-surfaces'):
        os.mkdir(dir +  '/mesh-complete/mesh-surfaces')
    if not os.path.exists(dir + '/centerlines'):
        os.mkdir(dir +  '/centerlines')

    max_area_cap = get_inlet_cap(mesher, walls)
    print("Max Area Cap: ")
    print(max_area_cap)
    out_caps = faces[1:]
    out_caps.remove(max_area_cap)
    np.save(dir+"/max_area_cap", np.asarray([max_area_cap]), allow_pickle = True)

    model.write(dir +  '/mesh-complete'+os.sep+'model_tmp','vtp')
    mesher.write_mesh(dir +  '/mesh-complete'+os.sep+'mesh-complete.mesh.vtu')
    mesh_out = modeling.PolyData()
    mesh_out.set_surface(mesher.get_surface())
    mesh_out.write(dir +  '/mesh-complete'+os.sep+'mesh-complete','exterior.vtp')
    mesh_out.set_surface(mesher.get_face_polydata(walls[0]))
    mesh_out.write(dir +  '/mesh-complete'+os.sep+'walls_combined','vtp')
    for face in mesher.get_model_face_ids():
        if face == walls[0]:
            continue
        mesh_out.set_surface(mesher.get_face_polydata(face))
        mesh_out.write(dir +  '/mesh-complete/mesh-surfaces'+os.sep+'cap_{}'.format(face),'vtp')

    cent_solid = modeling.PolyData()
    cent = vmtk.centerlines(model.get_polydata(), inlet_ids = [max_area_cap], outlet_ids = out_caps, use_face_ids = True)
    print("Centerlines generated.")
    cent_solid.set_surface(cent)
    print("Centerline surface set.")
    cent_solid.write(dir +  '/centerlines/centerline', "vtp")

    return
