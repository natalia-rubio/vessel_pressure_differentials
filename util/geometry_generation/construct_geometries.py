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

def generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs, sphere_ref = 2):

    segmentations = get_vessel_segmentations(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy, set_type, mesh_divs, sphere_ref)
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
            if anatomy == "curved" or anatomy == "straight":
                generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = 2)
            else:
                print("Didn't recognize anatomy type.")
    return

def launch_mesh_sweep(anatomy, set_type, num_geos = 1):

    dir = "data/synthetic_vessels/" + anatomy + "/" + set_type
    geos = os.listdir(dir)
    geos.sort()

    mesh_divs_list_curved = [1,2,3,4]
    sphere_ref_list = [1,2,3,4]
    for i in range(len(geos)):

        geo_name = geos[i]
        if not geo_name[0].isalnum():
            continue
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            print("Generating Geometry %d"%i)
            geo_params = load_dict(dir+"/"+geo_name+"/vessel_params_dict")
            print(geo_params)
            if anatomy == "curved" or anatomy == "straight":
                generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = mesh_divs_list_curved[i])
            if anatomy == "stenosed":
                generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = 3, sphere_ref = sphere_ref_list[i] )
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
#     generate_geometries(anatomy = "curved", set_type = "mesh_convergence", num_geos = 150)

if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    generate_geometries(anatomy = anatomy, set_type = set_type, num_geos = 150)
    # geo_params = load_dict("/Users/natalia/Desktop/vessel_pressure_differentials/data/synthetic_vessels/test/test/vertical_working/vessel_params_dict")

    # generate_vessel_mesh("vertical_not_working", geo_params, "test", "test", mesh_divs = 2)

# USE THIS COMMAND TO RUN WITH SIMVASCULAR PYTHON:
# /usr/local/sv/simvascular/2023-02-02/simvascular --python -- util/geometry_generation/launch_anatomy_sweep.py
# /Applications/SimVascular.app/Contents/Resources/simvascular --python -- util/geometry_generation/construct_geometries.py