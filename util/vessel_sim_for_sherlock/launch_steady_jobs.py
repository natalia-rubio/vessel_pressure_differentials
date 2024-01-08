from sherlock_util import *
from write_solver_files_steady import *

anatomy = sys.argv[1]
set_type = sys.argv[2]
num_time_steps = int(sys.argv[3])
num_cores = int(sys.argv[4])
num_geos = int(sys.argv[5])
num_flows = int(sys.argv[6])

num_launched = 0
print(f"Launching {num_geos} steady flow sweeps.")
dir = f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{set_type}"
geos = os.listdir(dir); geos.sort(); geo_ind = 0;#

while num_launched < num_geos:

    geo = geos[geo_ind]; geo_name = geo; print(geo_name)
    geo_ind += 1

    if not check_geo_name(geo):
        continue
    if not check_for_centerline(anatomy, set_type, geo_name):
        continue
    try:
        inlet_cap_number, cap_numbers = get_cap_info(anatomy, set_type, geo_name, correct_cap_numbers = 2)
    except:
        continue
    try:
        inlet_flow, inlet_area = load_params_dict(anatomy, set_type, geo_name)
    except:
        print(f"Couldn't find parameter dictionary.")
        continue

    for i, inlet_flow_fac in enumerate([0.25, 0.5, 0.75, 1]):

        try:
            flow_index = i; flow_name = f"flow_{flow_index}"
            if num_flows == 2:
                if i == 0 or i == 2:
                    continue
            if os.path.exists(f"/scratch/users/nrubio/synthetic_vessels_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{i}_red_sol"):
                print(f"Simulation already complete for flow {flow_index}")
                continue
            set_up_sim_directories(anatomy, set_type, geo_name, flow_name)
            flow_params = {"flow_amp": inlet_flow*inlet_flow_fac,
                            "vel_in": inlet_flow*inlet_flow_fac/inlet_area,
                            "res_1": 100,
                            "res_2": 100}

            time_step_size = 0.0001
            write_svpre_steady(anatomy, set_type, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size)
            write_inp_steady(anatomy, set_type, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size)
            write_flow_steady(anatomy, set_type, geo_name, flow_index, flow_params["flow_amp"], inlet_cap_number, num_time_steps, time_step_size)
            print("Done writing solver files.")
            f = open(f"/scratch/users/nrubio/synthetic_vessels/{anatomy}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()

            print(f"Started job for {geo} flow {flow_index}")
            write_geo_job(anatomy, set_type, geo, flow_name = flow_name, flow_index = flow_index, num_cores = num_cores, num_time_steps = num_time_steps)
            os.system(f"sbatch /scratch/users/nrubio/job_scripts/{geo[0]}_f{i}.sh")

        except:
            continue
    num_launched +=1
