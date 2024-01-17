import sys
sys.path.append("/Users/natalia/Desktop/vessel_pressure_differentials")
from util.geometry_generation.segmentation import *
from util.geometry_generation.modeling_and_meshing import *

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_unif_random(stats_list, lower_rng_bound = 0.4, upper_rng_bound = 0.6):
    #stats_list = [min, max, avg, std]
    return stats_list[0] + (lower_rng_bound + \
                            (upper_rng_bound-lower_rng_bound) * np.random.default_rng(seed=0).random()) * \
                            (stats_list[1]-stats_list[0])

def get_middle(stats_list):
    return stats_list[0] + 0.5 * (stats_list[1]-stats_list[0])

def generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs):

    segmentations = get_vessel_segmentations(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy, set_type, mesh_divs)
    print("Mesh Done!")
    return

def launch_anatomy_geo_sweep(anatomy, set_type, num_geos = 5):

    dir = "data/synthetic_vessels/"+anatomy+"/"+set_type
    geos = os.listdir(dir)
    for i in range(num_geos):

        geo_name = geos[i]
        print(geo_name)
        if not geo_name[0].isalnum():
            continue
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:
            print("Generating Geometry " + geo_name)
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/vessel_params_dict")

            print(geo_params)
            if anatomy == "basic":
                generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = 0.1)
            else:
                print("Didn't recognize anatomy type.")
    return

def launch_mesh_sweep(anatomy, set_type, num_geos = 1):

    dir = "data/synthetic_vessels/" + anatomy + "/" + set_type
    geos = os.listdir(dir)
    geos.sort()

    mesh_divs_list_basic = [1,2,3,4]
    for i in range(len(geos)):

        geo_name = geos[i]
        if not geo_name[0].isalnum():
            continue
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            print("Generating Geometry %d"%i)
            geo_params = load_dict(dir+"/"+geo_name+"/vessel_params_dict")
            print(geo_params)
            if anatomy == "basic" or anatomy == "straight":

                generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = mesh_divs_list_basic[i])
            else:
                print("Didn't recognize anatomy type.")

    return

def generate_geometries(anatomy, set_type, num_geos):
    if set_type == "mesh_convergence":
        launch_mesh_sweep(anatomy, set_type, num_geos)
    elif set_type == "random":
        launch_anatomy_geo_sweep(anatomy, set_type, num_geos)
    return

# if __name__ == "__main__":
#     generate_geometries(anatomy = "basic", set_type = "mesh_convergence", num_geos = 150)

if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    generate_geometries(anatomy = "straight", set_type = "mesh_convergence", num_geos = 150)

# USE THIS COMMAND TO RUN WITH SIMVASCULAR PYTHON:
# /usr/local/sv/simvascular/2023-02-02/simvascular --python -- util/geometry_generation/launch_anatomy_sweep.py
# /Applications/SimVascular.app/Contents/Resources/simvascular --python -- util/geometry_generation/construct_geometries.py