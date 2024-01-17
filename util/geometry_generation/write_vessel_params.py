import sys
sys.path.append("/Users/natalia/Desktop/vessel_pressure_differentials")
from util.tools.basic import *

def write_anatomy_junctions(anatomy, set_type, num_junctions):

    if not os.path.exists("data/synthetic_vessels"):
        os.mkdir("data/synthetic_vessels")
    if not os.path.exists(f"data/synthetic_vessels/{anatomy}"):
        os.mkdir(f"data/synthetic_vessels/{anatomy}")
    if not os.path.exists(f"data/synthetic_vessels/{anatomy}/{set_type}"):
        os.mkdir(f"data/synthetic_vessels/{anatomy}/{set_type}")

    for i in range(num_junctions):
        junction_name = f"{anatomy[0]}_{i}"
        if os.path.exists(f"data/synthetic_vessels/{anatomy}/{set_type}/{junction_name}/vessel_params_dict") == False:
                if not os.path.exists(f"data/synthetic_vessels/{anatomy}/{set_type}/{junction_name}"):
                    os.mkdir(f"data/synthetic_vessels/{anatomy}/{set_type}/{junction_name}")

                print(f"Generating {junction_name}")
                if set_type == "mesh_convergence":
                    params_rand = generate_nominal_params(anatomy)

                elif set_type == "random":
                     continue
                
                save_dict(params_rand, f"data/synthetic_vessels/{anatomy}/{set_type}/{junction_name}/vessel_params_dict")
    return

def generate_nominal_params(anatomy):
    params_stat_dict = load_dict(f"data/param_stat_dicts/param_stat_dict_{anatomy}")
    nominal_params = {}
    for param in params_stat_dict.keys():
        nominal_params.update({param : get_middle(params_stat_dict[param])})
    return nominal_params

def generate_param_stat_dicts():
    anatomy = "basic"
    params_stat_dict = {"length": [8, 12, 10, 1], 
                "curvature": [30, 90, 60, 20],
                "inlet_radius": [0.2, 0.6, 0.4, 0.1],
                "outlet_radius": [0.2, 0.6, 0.4, 0.1],
                "stenosis_magnitude": [-0.5, 0.5, 0, 0.2], 
                "stenosis_spread": [0.05, 0.3, 0.15, 0.1],
                "stenosis_location": [0, 1, 0.5, 0.2],
                "inlet_velocity": [50, 150, 100, 20]}
    
    save_dict(params_stat_dict, f"data/param_stat_dicts/param_stat_dict_{anatomy}")

    anatomy = "straight"
    params_stat_dict = {"length": [8, 12, 10, 1], 
                "curvature": [0, 0, 0, 0],
                "inlet_radius": [0.2, 0.6, 0.4, 0.1],
                "outlet_radius": [0.2, 0.6, 0.4, 0.1],
                "stenosis_magnitude": [-0.5, 0.5, 0, 0.2], 
                "stenosis_spread": [0.05, 0.3, 0.15, 0.1],
                "stenosis_location": [0, 1, 0.5, 0.2],
                "inlet_velocity": [50, 150, 100, 20]}
    
    save_dict(params_stat_dict, f"data/param_stat_dicts/param_stat_dict_{anatomy}")
    return
# params_stat_dict = load_dict("data/param_stat_dict")[anatomy_name]
# print(params_stat_dict)

if __name__ == '__main__':
    generate_param_stat_dicts()
    write_anatomy_junctions(anatomy = "basic", set_type = "mesh_convergence", num_junctions = 4)
    write_anatomy_junctions(anatomy = "straight", set_type = "mesh_convergence", num_junctions = 4)