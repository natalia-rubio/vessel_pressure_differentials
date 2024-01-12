import os
import sys
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb
import random
import copy
from scipy import interpolate
import pickle
import dgl
import tensorflow as tf
from util.junction_extraction_util.get_avg_sol import *

np.seterr(all='raise')
np.set_printoptions(threshold=sys.maxsize)

def get_inds(arr, vals):
    arr = np.asarray(arr); vals = np.asarray(vals)
    inds = 0 * vals
    for i in range(vals.size):
        inds[i] = np.where(arr == vals[i])[0][0]
    inds = list(inds)
    return inds

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res

def get_all_arrays(geo):
    # collect all arrays
    #cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())
    return point_data

def read_solution(geo_name, flow_name, dir = "/home/nrubio/Desktop/synthetic_junctions"):
    home_dir = os.path.expanduser("~")
    fname = f"{dir}/{geo_name}/avg_solution_flow_{flow_name}.vtu"
    print(fname)
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def read_centerline(geo_name, dir = "/home/nrubio/Desktop/centerlines"):
    home_dir = os.path.expanduser("~")
    fname = f"{dir}/{geo_name}/centerline.vtp"

    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def augment_time(field, times_before, aug_factor):
    ntimes_new = aug_factor*len(times_before)
    end_time = max(times_before)
    start_time = min(times_before)
    n_points = field.shape[1]
    times_new = np.linspace(start_time, end_time, ntimes_new, endpoint=True)

    field_new = np.zeros((ntimes_new, n_points))
    field_new_der = np.zeros((ntimes_new, n_points))
    field_new_der2 = np.zeros((ntimes_new, n_points))
    for point_i in range(n_points):
      y = field[:, point_i]
      tck = interpolate.splrep(times_before, y, s=0)
      field_new[:,point_i] = interpolate.splev(times_new, tck, der=0)
      field_new_der[:,point_i] = interpolate.splev(times_new, tck, der=1)
      field_new_der2[:,point_i] = interpolate.splev(times_new, tck, der=2)

    return field_new, field_new_der, field_new_der2

def load_centerline_data(geo_name, dir):
    cent = read_centerline(geo_name, dir).GetOutput()
    cent_array = get_all_arrays(cent)

    #Extract Geometry ----------------------------------------------------
    pt_id = cent_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = cent_array["BranchId"].astype(int)
    junction_id = cent_array["BifurcationId"].astype(int)
    axial_distance = cent_array["Path"]  # distance along centerline
    area = cent_array["CenterlineSectionArea"]
    direction = cent_array["CenterlineSectionNormal"]  # vector normal direction
    direction_norm = np.linalg.norm( direction, axis=1, keepdims=True)  # norm of direction vector
    direction = np.transpose(np.divide(direction,direction_norm))  # normalized direction vector
    angle1 = direction[0,:].reshape(-1,)
    angle2 = direction[1,:].reshape(-1,)
    angle3 = direction[2,:].reshape(-1,)
    return pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3

def identify_junctions(junction_id, branch_id, pt_id):
    junction_ids = np.linspace(0,max(junction_id),max(junction_id)+1).astype(int)
    branch_ids = np.linspace(0,max(branch_id),max(branch_id)+1).astype(int)
    junction_dict = {}
    for i in junction_ids:

        junction_pts = pt_id[junction_id == i] # find all points in junction
        branch_pts_junc = [] # inlet and outlet point ids of junction
        branch_ids_junc = [] # branch ids of junction
        branch_pts_junc.append(max([min(junction_pts)-1, 0])) # find "inlet" of junction (point with smallest Id)
        branch_ids_junc.append(branch_id[pt_id == min(junction_pts)-1][0]) # find the branch to which the inlet belongs
        branch_counter = 1 # initialize counter for the number of branches
        # loop over all branches in model
        for j in branch_ids:
            branch_pts = pt_id[branch_id == j] # find points belonging to branch
            shared_pts = np.intersect1d(junction_pts+1, branch_pts) # find points adjacent to the junction
            # if there is an adjacent point
            if len(shared_pts) != 0 and j not in branch_ids_junc : # if there is a shared point in the branch
                branch_counter = branch_counter + 1 # increment branch counter
                branch_ids_junc.append(j.astype(int)) # add outlet branch Id to outlet branch array
                branch_pts_junc.append(min([min(branch_pts).astype(int), max(branch_pts)])) # add outlet point Id to outlet point array
        junction_dict.update({i : branch_pts_junc})
        #assert i == 0, "There should only be one junction,"
    return junction_dict

def get_junction_pts(flow, junction_id, junction_dict):

    inlets = []; outlets = [] # initialize inlet and outlet list
    branch_pts_junc = junction_dict[junction_id] #
    return branch_pts_junc

def load_soln_data(soln_dict):

    pressure_in_time = soln_dict["pressure_in_time"]
    flow_in_time = soln_dict["flow_in_time"]
    times = soln_dict["times"]
    time_interval = 0.1

    return pressure_in_time, flow_in_time, times, time_interval

def load_vmr_model_data(model, results_dir):
    time_interval = 0.1
    soln = read_geo(f"{results_dir}/{model}").GetOutput()  # get 3d flow data
    soln_array = get_all_arrays(soln)
    #Extract Geometry ----------------------------------------------------
    pt_id = soln_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = soln_array["BranchId"].astype(int)
    junction_id = soln_array["BifurcationId"].astype(int)
    axial_distance = soln_array["Path"]  # distance along centerline
    area = soln_array["area"]
    direction = soln_array["CenterlineSectionNormal"]  # vector normal direction
    direction_norm = np.linalg.norm( direction, axis=1, keepdims=True)  # norm of direction vector
    direction = np.transpose(np.divide(direction,direction_norm))  # normalized direction vector
    angle1 = direction[0,:].reshape(-1,)
    angle2 = direction[1,:].reshape(-1,)
    angle3 = direction[2,:].reshape(-1,)

    # Extract timesteps and pressures + velocities at each timestep -------
    pressure_in_time= np.zeros((0,num_pts)) # initialize pressure_in_time matrix, each column is a mesh point, each row is a timestep
    flow_in_time = np.zeros((0,num_pts)) # initialize flow_in_time matrix, each column is a mesh point, each row is a timestep
    times = list()  # list of timesteps
    for key in soln_array.keys():
        if key[0:8] == "pressure":
            pressure_in_time = np.vstack((pressure_in_time, soln_array[key]))  # add timestep row to pressure_in_time
            times.append(float(key[9:17]))  # add timestep to times

        elif key[0:8] == "velocity": # NOTE: keys are labeled "velocity" but are actually flow!!!
            flow_in_time= np.vstack((
                flow_in_time, soln_array[key]))  # add timestep column to flow_in_time

    return pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, pressure_in_time, flow_in_time, times, time_interval

def classify_branches(flow, junc_pts, pt_arr):
    inlets = []; outlets = [] # initialize inlet and outlet list
    for branch_pt in junc_pts:
      if branch_pt == min(junc_pts) and flow[np.isin(pt_arr, branch_pt)] > 0:
          inlets.append(branch_pt)
      elif branch_pt == min(junc_pts) and flow[np.isin(pt_arr, branch_pt)] < 0:
          outlets.append(branch_pt)
      elif flow[np.isin(pt_arr, branch_pt)] > 0:
          outlets.append(branch_pt)
      elif flow[np.isin(pt_arr, branch_pt)] < 0:
          inlets.append(branch_pt)
      else:
          outlets.append(branch_pt)
    return inlets, outlets

def get_inlet_outlet_pairs(num_inlets, num_outlets):

    inlet_list = []; outlet_list = []
    for inlet in range(num_inlets):
        for outlet in range(num_outlets):
            inlet_list.append(inlet); outlet_list.append(outlet)
    inlet_outlet_pairs = (tf.convert_to_tensor(inlet_list, dtype=tf.int32),
                            tf.convert_to_tensor(outlet_list, dtype=tf.int32))
    return inlet_outlet_pairs

def get_outlet_pairs(num_outlets):
    outlet_list1 = []; outlet_list2 = []
    for outlet1 in range(num_outlets):
        for outlet2 in range(num_outlets):
            outlet_list1.append(outlet1); outlet_list2.append(outlet2)
    outlet_pairs = (tf.convert_to_tensor(outlet_list1, dtype=tf.int32),
                            tf.convert_to_tensor(outlet_list2, dtype=tf.int32))
    return outlet_pairs

def get_angle_diff(angle1, angle2):
    try:
        angle_diff = np.arccos(np.dot(angle1, angle2))
    except:
        pdb.set_trace()
    return angle_diff

def get_flow_hist(flow_in_time_aug, time_index, num_time_steps_model):
    if time_index == np.linspace(0, num_time_steps_model-1, num_time_steps_model).astype(int)[0]:
      flow_hist1 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
      flow_hist2 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
      flow_hist3 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
    elif time_index == np.linspace(0, num_time_steps_model-1, num_time_steps_model).astype(int)[1]:
      flow_hist1 = flow_in_time_aug[time_index-1, :] # velocity at timestep
      flow_hist2 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
      flow_hist3 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
    elif time_index == np.linspace(0, num_time_steps_model-1, num_time_steps_model).astype(int)[2]:
      flow_hist1 = flow_in_time_aug[time_index-1, :] # velocity at timestep
      flow_hist2 = flow_in_time_aug[time_index-2, :] # velocity at timestep
      flow_hist3 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
    else:
      flow_hist1 = flow_in_time_aug[time_index-1, :] # velocity at timestep
      flow_hist2 = flow_in_time_aug[time_index-2, :] # velocity at timestep
      flow_hist3 = flow_in_time_aug[time_index-3, :] # velocity at timestep
    return flow_hist1, flow_hist2, flow_hist3

def scale(scaling_dict, field, field_name):
    mean = scaling_dict[field_name][0]; std = scaling_dict[field_name][1]
    scaled_field = (field-mean)/std
    return scaled_field
