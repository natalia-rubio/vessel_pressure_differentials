import os
import sys
import math
import numpy as np
import pickle
import shutil
import pdb
import subprocess
import time
import copy
#from get_scalers_synthetic import *
from centerline_proj import *

def check_convergence(anatomy, set_type, geo_name, flow_index, num_time_steps):
    flow_name = f"flow_{flow_index}"
    results_dir = f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{flow_index}_red_sol"
    centerline_dir = f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"

    print("Averaging 3D results.")

    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = centerline_dir)

    conv = True
    # soln_dict, conv = get_avg_steady_results(fpath_1d = centerline_dir,
    #                 fpath_3d = f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}.vtu",
    #                 fpath_out = results_dir,
    #                 pt_inds = junc_pt_ids, only_caps=False)

    if conv == True:
        print("Converged!"); geometry = geo_name; flow = flow_index
        fpath_1d = f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geometry}/centerlines/centerline.vtp"
        fpath_3d = f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}/{geometry}/flow_{flow}/solution_flow_{flow}.vtu"
        fpath_out = f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geometry}/1dsol_flow_solution_{flow}.vtp"
        extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = num_time_steps)
        sys.exit(1)


    return

anatomy = sys.argv[1]; set_type = sys.argv[2]; geo_name = sys.argv[3]; flow_index = sys.argv[4]; num_time_steps = sys.argv[5]
check_convergence(anatomy = anatomy, set_type= set_type, geo_name = geo_name, flow_index = flow_index, num_time_steps = num_time_steps)
