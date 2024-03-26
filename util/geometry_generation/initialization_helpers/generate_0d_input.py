# from tools.vtk_functions import *
# from tools.vtk_to_xdmf import *

'''Test the ROM simulation class. 

   Generate a reduced-order modeling (ROM) simulation input file.

   Use the centerlines and faces from the Demo project.
'''
import os
from pathlib import Path
import sv
import sys
import vtk
import numpy as np
def get_cap_numbers(cap_dir):
    file_names = os.listdir(cap_dir)
    cap_numbers = []
    print(file_names)
    print(cap_dir)
    for cap_file in file_names:
        if cap_file[0:3] == "cap":
            cap_numbers.append(cap_file[0:-4])
    return cap_numbers

def write_0D_flow(anatomy, set_type, geo_name, flow_index, flow_amp):
    flow_name = "flow_" + str(flow_index)
    flow = ""
    num_time_steps = 5
    t = np.linspace(start = 0, stop = num_time_steps, num = num_time_steps)
    q = t*0
    for i in range(t.size):
        q[i] = flow_amp * (flow_index + 1) * 0.25

        flow = flow + "%1.5f %1.3f\n" %(i*0.2, q[i])
    f = open("data/synthetic_vessels/" + anatomy + "/" + set_type + "/" + geo_name + "/" + flow_name + "/" + flow_name + "_zerod.flow", "w")
    f.write(flow)
    f.close()
    return

def write_input_file(anatomy, set_type, geo_name, flow_index):
    geo_dir = "data/synthetic_vessels/" + anatomy + "/" + set_type + "/" + geo_name
    flow_name = "flow_" + str(flow_index)

    inlet_cap_number = int(np.load(geo_dir + "/max_area_cap.npy")[0])
    cap_numbers = get_cap_numbers(geo_dir + "/mesh-complete/mesh-surfaces/"); print(cap_numbers)
    cap_numbers.remove(('cap_' + str(inlet_cap_number)))

    ## Create a ROM simulation.
    input_dir = str(geo_dir + "/" + "input")
    rom_simulation = sv.simulation.ROM() 

    ## Create ROM simulation parameters.
    params = sv.simulation.ROMParameters()

    ## Mesh parameters.
    mesh_params = params.MeshParameters()

    ## Model parameters.
    model_params = params.ModelParameters()
    model_params.name = "synthetic_vessel"
    model_params.inlet_face_names = ['cap_' + str(inlet_cap_number)] 
    model_params.outlet_face_names = cap_numbers
    model_params.centerlines_file_name = geo_dir + '/centerlines/centerline.vtp' 

    ## Fluid properties.
    fluid_props = params.FluidProperties()
    material = params.WallProperties.OlufsenMaterial()
    print("Material model: {0:s}".format(str(material)))

    ## Set boundary conditions.
    bcs = params.BoundaryConditions()
    #bcs.add_resistance(face_name='outlet', resistance=1333)
    bcs.add_velocities(face_name='cap_' + str(inlet_cap_number), file_name= geo_dir + "/" + flow_name + "/" + flow_name + "_zerod.flow")
    bcs.add_resistance(face_name=cap_numbers[0], resistance=100.0)
    #bcs.add_resistance(face_name=cap_numbers[1], resistance=100.0)

    ## Set solution parameters. 
    #
    solution_params = params.Solution()
    solution_params.time_step = 0.2
    solution_params.num_time_steps = 5

    ## Write a 1D solver input file.
    #
    if not os.path.exists(geo_dir + "/" + flow_name + "/" + "zerod_files"):
        os.mkdir(geo_dir + "/" + flow_name + "/" + "zerod_files")
    output_dir = geo_dir + "/" + flow_name + "/" + "zerod_files"
    rom_simulation.write_input_file(model_order=0, model=model_params, mesh=mesh_params, fluid=fluid_props, material=material, boundary_conditions=bcs, solution=solution_params, directory=output_dir)
    return