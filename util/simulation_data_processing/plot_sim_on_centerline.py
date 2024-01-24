import sys
sys.path.append("/Users/natalia/Desktop/vessel_pressure_differentials")
from util.tools.basic import *
from util.tools.vtk_functions import *
import vtk
import os
import numpy as np
plt.rcParams['font.size'] = 8

def extract_soln(fpath_1dsol, offset = 10):
    soln = read_geo(fpath_1dsol).GetOutput()  # get 3d flow data
    soln_array = get_all_arrays(soln)[0]
    print(soln_array.keys())
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

    flow = soln_array["velocity_00500"]
    pressure = soln_array["pressure_00500"]
    area = soln_array["area"]
    radius = np.sqrt(area/np.pi)
    pressure_poiseuille = [pressure[offset]]
    for i in range(len(incs)):
        ind = i + offset
        pressure_poiseuille.append(pressure_poiseuille[i] - dP_poiseuille(flow[ind], radius[ind], incs[ind]))
    pressure_poiseuille = np.asarray(pressure_poiseuille)
    return dist_array, pressure, pressure_poiseuille, flow, area

def plot_vars(anatomy, set_type, geometry, flow_name, plot_pressure = True):
    if anatomy == "straight":
        cell_list = ["4.1E3", "1.9E4", "5.4E4", "1.2E5"]
    elif anatomy == "curved":
        cell_list = ["3.8E3", "2.0E4", "5.5E4", "1.2E5"]
    offset = 0
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3,1, 1]}, sharex=True)
    geo_list = os.listdir(f"data/synthetic_vessels_reduced_results/{anatomy}/{set_type}")
    geo_list.sort()
    for geo in geo_list:
        if not geo[0].isalnum():
            geo_list.remove(geo)
    print(geo_list)
    for geometry in geo_list:

        fpath_1dsol = f"data/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geometry}/1dsol_flow_solution_{flow_name}.vtp"
        dist_array, pressure, pressure_poiseuille, flow, area = extract_soln(fpath_1dsol, offset)
        
        ax1.plot(dist_array, pressure/1333, linewidth = 1, label = f"Simulation ({cell_list[geo_list.index(geometry)]} cells)")
        ax1.set_ylabel("Pressure (mmHg)")
        
        ax2.plot(dist_array, flow, linewidth = 1)
        ax2.set_ylim([0, 1.5*np.max(flow)])
        ax2.set_ylabel("Flow (cm/s)")

        ax3.plot(dist_array, area, linewidth = 1)
        ax3.set_ylabel("Area (cm^2)")
        ax3.set_xlabel("Distance (cm)")
        ax3.set_ylim([0, 1.5*np.max(area)])
    ax1.plot(dist_array, pressure_poiseuille/1333, "k", linewidth = 1, linestyle = "--", label = "Poiseuille")
    ax1.legend()
    fig.savefig(f"results/simulation_centerline_results/{anatomy}_{set_type}_flow_{flow_name}.pdf")

    return

if __name__ == "__main__":
    anatomy = "curved"
    set_type = "mesh_convergence"
    geometry = "s_3"
    flow = 1
    plot_vars(anatomy, set_type, geometry, flow)
