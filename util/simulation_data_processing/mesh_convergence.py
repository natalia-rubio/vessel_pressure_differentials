import sys
sys.path.append("/Users/natalia/Desktop/vessel_pressure_differentials")
from util.tools.basic import *
from util.tools.vtk_functions import *
import vtk
import os
import numpy as np

def make_mesh_convergence_plot(anatomy, set_type):
    geos = os.listdir(f"data/synthetic_vessels_reduced_results/{anatomy}/{set_type}")
    for geo in geos:
        if not geo[0].isalnum():
            geos.remove(geo)
    geos.sort()
    print(geos)
    plt.clf()
    if anatomy == "straight":
        cell_list = ["4.1E3", "1.9E4", "5.4E4", "1.2E5"]
    elif anatomy == "curved":
        cell_list = ["3.8E3", "2.0E4", "5.5E4", "1.2E5"]
    for geo in geos:
        results_dir = f"data/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geo}/"
        dP_list = []; flow_list = []
        for i in range(4):
            try:
                flow_result_dir = results_dir + f"flow_{i}_red_sol"
                soln_dict = load_dict(flow_result_dir)
                dP_list.append((soln_dict["pressure_in_time"][1,1] - soln_dict["pressure_in_time"][1,0])/1333)
                flow_list.append(soln_dict["flow_in_time"][1,1])
            except:
                continue
        plt.scatter(flow_list, dP_list, label = f"{cell_list[geos.index(geo)]} cells")
    plt.xlabel("Flow (cm/s)")
    plt.ylabel("Pressure Difference (mmHg)")
    plt.legend()
    #pressure_poiseuille = [dP_poiseuille(np.asarray(flow_list), 0.4, 0.1)]
    #plt.plot(np.asarray(flow_list), pressure_poiseuille/1333, "k", linewidth = 1, linestyle = "--", label = "Poiseuille")
    plt.savefig(f"results/mesh_convergence_{anatomy}.png", dpi = 300)

if __name__ == "__main__":
    anatomy = "curved"
    set_type = "mesh_convergence"
    make_mesh_convergence_plot(anatomy, set_type)