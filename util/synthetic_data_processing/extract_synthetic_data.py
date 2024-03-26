import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def extract_steady_flow_data(anatomy, geo, require4):
    results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{geo}/"

    flow_lists = [[0],[0]]
    dP_lists = [[0],[0]]
    dP_junc_lists = [[0],[0]]

    for i in range(4):
        # if i == 1 or i ==3:
        #     continue
        try:
            flow_result_dir = results_dir + f"flow_{i}_red_sol"

            if not os.path.exists(results_dir):
                if require4:
                    assert os.path.exists(results_dir), f"Flow {i} missing for geometry {geo}"
                else:
                    continue
            soln_dict = load_dict(flow_result_dir)

            for outlet_ind in range(2):

                flow_lists[outlet_ind] += [soln_dict["flow"][outlet_ind]]
                dP_lists[outlet_ind] += [soln_dict["dp_end"][outlet_ind]]
                dP_junc_lists[outlet_ind] += [soln_dict["dp_junc"][outlet_ind]]
        except:
            if require4:
                raise ValueError
    return flow_lists, dP_lists, dP_junc_lists

def extract_unsteady_flow_data(anatomy, geo):
    unsteady_result_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{geo}/unsteady_red_sol"
    unsteady_soln_dict = load_dict(unsteady_result_dir)

    if anatomy[0:5] == "Aorta":
        print("Aorta")
        posi_Q_ind = np.all(unsteady_soln_dict["flow_in_time"] > 0, axis = 1)
        unsteady_soln_dict["flow_in_time"] = unsteady_soln_dict["flow_in_time"][posi_Q_ind,:]
        unsteady_soln_dict["pressure_in_time"] = unsteady_soln_dict["pressure_in_time"][posi_Q_ind,:]
        unsteady_soln_dict["flow_in_time"] = unsteady_soln_dict["flow_in_time"][:80, :]
        unsteady_soln_dict["pressure_in_time"] = unsteady_soln_dict["pressure_in_time"][:80, :]
    if anatomy[0:5] == "Pulmo":
        #import pdb; pdb.set_trace()
        unsteady_soln_dict["flow_in_time"] = unsteady_soln_dict["flow_in_time"][105:-10, :]
        unsteady_soln_dict["pressure_in_time"] = unsteady_soln_dict["pressure_in_time"][105:-10, :]

        if np.any(unsteady_soln_dict["flow_in_time"]<0):
            posi_Q_ind = np.all(unsteady_soln_dict["flow_in_time"] > 0, axis = 1)
            print(posi_Q_ind)
            import pdb; pdb.set_trace()
    dQdt_unsteady = (unsteady_soln_dict["flow_in_time"][1:,1:] - unsteady_soln_dict["flow_in_time"][:-1,1:])/0.002
    Q_unsteady = unsteady_soln_dict["flow_in_time"][1:,1:]
    dP_unsteady = unsteady_soln_dict["pressure_in_time"][1:, 1:] - unsteady_soln_dict["pressure_in_time"][1:, 0].reshape(-1, 1)
    if np.max(dP_unsteady) < 10:
        import pdb; pdb.set_trace()

    unsteady_flow_lists = [[0],[0]]
    unsteady_flow_der_lists = [[0],[0]]
    unsteady_dP_lists = [[0],[0]]

    for outlet_ind in range(2):
        unsteady_dP_lists[outlet_ind] = dP_unsteady[:, outlet_ind]
        unsteady_flow_lists[outlet_ind] = Q_unsteady[:, outlet_ind]
        unsteady_flow_der_lists[outlet_ind] = dQdt_unsteady[:, outlet_ind]
    return unsteady_flow_lists, unsteady_flow_der_lists, unsteady_dP_lists



def collect_synthetic_results(anatomy, require4 = True, unsteady = False):
    print(anatomy)

    char_val_dict = {"outlet_radius": [],
                    "inlet_area": [],
                    "inlet_radius": [],
                    "outlet_area": [],
                    "angle": [],
                    "flow_list": [],
                    "unsteady_flow_list": [],
                    "unsteady_flow_der_list": [],
                    "dP_list": [],
                    "unsteady_dP_list": [],
                    "dP_junc_list": [],
                    "inlet_length": [],
                    "outlet_length": [],
                    "name": []}

    home_dir = os.path.expanduser("~")
    geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}"); geos.sort(); print(geos)
    plt.clf()
    for j, geo in enumerate(geos[0:]):

        try:
            #import pdb; pdb.set_trace()
            junction_params = load_dict(f"data/synthetic_junctions/{anatomy}/{geo}/junction_params_dict")

            flow_lists, dP_lists, dP_junc_lists = extract_steady_flow_data(anatomy, geo, require4)

            if len(flow_lists[0]) <= 2:
                continue

            if unsteady:
                try:
                    unsteady_flow_lists, unsteady_flow_der_lists, unsteady_dP_lists = extract_unsteady_flow_data(anatomy, geo)
                    char_val_dict["unsteady_flow_list"] += unsteady_flow_lists
                    char_val_dict["unsteady_flow_der_list"] += unsteady_flow_der_lists
                    char_val_dict["unsteady_dP_list"] += unsteady_dP_lists
                except:
                    continue
            print(flow_lists, dP_lists)
            char_val_dict["flow_list"] += flow_lists
            char_val_dict["dP_list"] += dP_lists
            char_val_dict["dP_junc_list"] += dP_junc_lists




            results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{geo}/flow_1_red_sol"
            soln_dict = load_dict(results_dir)

            char_val_dict["inlet_radius"] += [np.sqrt(soln_dict["area"][2]/np.pi), np.sqrt(soln_dict["area"][2]/np.pi)]
            char_val_dict["outlet_radius"] += [np.sqrt(soln_dict["area"][0]/np.pi), np.sqrt(soln_dict["area"][1]/np.pi)]

            char_val_dict["inlet_area"] += [soln_dict["area"][2],soln_dict["area"][2]]
            char_val_dict["outlet_area"] += [soln_dict["area"][0], soln_dict["area"][1]]

            char_val_dict["inlet_length"] += [soln_dict["length"][2], soln_dict["length"][2]]
            char_val_dict["outlet_length"] += [soln_dict["length"][0], soln_dict["length"][1]]

            char_val_dict["angle"] += [junction_params["outlet1_angle"], junction_params["outlet2_angle"]]
            char_val_dict["name"] += [geo+"_1", geo+"_2"]

        except:
            print(f"Problem extracting junction data.  Skipping {geo}.")
            continue

    #import pdb; pdb.set_trace()


    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    print(f"Extracted {len(char_val_dict['name'])} Outlets")
    return
