import sys
sys.path.append("/Users/natalia/Desktop/vessel_pressure_differentials")
from util.tools.basic import *
from util.tools.vtk_functions import *
import vtk
import os
import numpy as np


def plot_vars(anatomy, set_type, geometry, flow, plot_pressure = True):
    offset = 0
    fpath_1dsol = f"data/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geometry}/1dsol_flow_solution_{flow}.vtp"
    soln = read_geo(fpath_1dsol).GetOutput()  # get 3d flow data

    soln_array = get_all_arrays(soln)[0]
    pts = v2n(soln.GetPoints().GetData())
    #Extract Geometry ----------------------------------------------------
    pt_id = soln_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = soln_array["BranchIdTmp"].astype(int)
    junction_id = soln_array["BifurcationId"].astype(int)


    locs = (pts)[np.argsort(pt_id)]
    assert np.all(locs == pts), "Points not sorted by id"
    dist = [0]
    incs = np.linalg.norm(locs[1:,:] - locs[0:-1, :], axis = 1)
    for i in range(len(incs)):
        dist.append(dist[i] + incs[i])
    dist_array = np.asarray(dist)

    flow = soln_array["velocity_00300"]
    pressure = soln_array["pressure_00300"]/1333.22
    area = soln_array["area"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(dist_array, pressure, "b")
    ax1.set_ylabel("Pressure (mmHg)")
    
    ax2.plot(dist_array, flow, "b")
    ax2.set_ylabel("Flow (cm/s)")

    ax3.plot(dist_array, area, "b")
    ax3.set_ylabel("Area (cm^2)")
    pdb.set_trace()
    plt.show()
    fig.savefig(f"results/simulation_centerline_results/{anatomy}_{set_type}_{geometry}_flow_{flow}.png")

    return

if __name__ == "__main__":
    anatomy = "basic"
    set_type = "mesh_convergence"
    geometry = "b_0"
    flow = 1
    plot_vars(anatomy, set_type, geometry, flow)
