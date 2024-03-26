from util.geometry_generation.initialization_helpers.generate_0d_input import *
from util.geometry_generation.initialization_helpers.projection import *
import numpy as np

def generate_initial_sol(geo_name, anatomy, set_type, params):
   for flow_index in range(4):

      flow_name = "flow_" + str(flow_index)
      dir = "data/synthetic_vessels/"+anatomy+"/"+set_type+"/"+geo_name+"/"+flow_name
      
      if not os.path.exists(dir):
          os.mkdir(dir)

      flow_amp = params["inlet_velocity"]*np.pi*params["inlet_radius"]**2
      write_0D_flow(anatomy, set_type, geo_name, flow_index, flow_amp)
      write_input_file(anatomy, set_type, geo_name, flow_index)
      print(r"sed -i '' 's/internal_junction/NORMAL_JUNCTION/g' " + dir + "/zerod_files/solver_0d.json")
      os.system(r"sed -i '' 's/internal_junction/NORMAL_JUNCTION/g' " + dir + "/zerod_files/solver_0d.json")
      os.system("source zerod_env/bin/activate\n" + \
                "/Users/natalia/miniforge3/envs/zerod/bin/svzerodsolver " + \
               dir + "/zerod_files/solver_0d.json " + \
               dir + "/zerod_files/zerod_soln.csv")
      #os.system("deactivate")
      
      project_0d_to_3D(anatomy, set_type, geo_name, flow_index)

   return