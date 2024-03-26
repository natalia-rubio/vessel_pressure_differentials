import os
import sys
import numpy as np
import time
import copy
import pickle
from geo_processing import *
import subprocess
import time
import copy

def set_up_sim_directories(anatomy, set_type, geo_name, flow_name, num_cores):
    print("Starting run_simulation function.")
    if not os.path.exists(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}"):
        os.mkdir(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}")
    if not os.path.exists(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}"):
        os.mkdir(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}")
    if not os.path.exists(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geo_name}"):
        os.mkdir(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geo_name}")
    results_dir = f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geo_name}"
    if os.path.exists(results_dir) == False:
        os.mkdir(results_dir)

    if not os.path.exists(f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/{flow_name}"):
        print("inital solutions missing")
    if os.path.exists(f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/{flow_name}/{num_cores}-procs_case"):
        os.system(f"rm -r /scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/{flow_name}/{num_cores}-procs_case")

    os.system(f"mkdir /scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/{flow_name}")
    return

def check_geo_name(geo):
    if not geo[0].isalnum():
        print("Not a valid geometry name")
        return False
    return True

def check_for_centerline(anatomy, set_type, geo_name):
    try:
        centerline_dir = f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"
        pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = centerline_dir)
        #junction_dict, junc_pt_ids = identify_junctions(junction_id, branch_id, pt_id)
    except:
        print(f"Couldn't process centerline.  Removing {geo_name}")
        os.system(f"rm -r /scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}")
        return False
    return True

def get_cap_info(anatomy, set_type, geo_name, correct_cap_numbers = 2):
    inlet_cap_number = int(np.load(f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/max_area_cap.npy")[0])
    cap_numbers = get_cap_numbers(f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/mesh-complete/mesh-surfaces/"); print(cap_numbers)

    if len(cap_numbers) != correct_cap_numbers:
        print("Wrong number of caps.  Deleting.")
        os.system(f"rm -r /scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}")
        ValueError("Wrong number of caps.")
    return inlet_cap_number, cap_numbers

def load_params_dict(anatomy, set_type, geo_name):
    params_dict = load_dict(f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/vessel_params_dict")
    inlet_velocity = params_dict["inlet_velocity"]
    inlet_area = np.pi * params_dict["inlet_radius"]**2
    inlet_flow = inlet_velocity * inlet_area
    return inlet_flow, inlet_area

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_cap_numbers(cap_dir):
    file_names = os.listdir(cap_dir)
    cap_numbers = []
    print(file_names)
    print(cap_dir)
    for cap_file in file_names:
        if cap_file[0:3] == "cap":
            cap_numbers.append(int(cap_file[4:-4]))
    return cap_numbers

