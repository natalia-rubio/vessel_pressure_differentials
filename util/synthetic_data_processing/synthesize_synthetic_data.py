import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def get_name_end( unsteady = False, use_steady_ab = True):
    name_end = ""
    if unsteady and use_steady_ab:
        name_end += "steady_ab"
    if not unsteady:
        name_end += "steady"

def get_coefs(anatomy, rm_low_r2 = True, unsteady = False, use_steady_ab = True):

    num_outlets = 2
    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    char_val_dict.update({"coef_a": [],
                    "coef_b": [],
                    "coef_L": [],
                    "coef_a_UO": [],
                    "coef_b_UO": [],
                    "coef_L_UO": []})
    to_rm = []
    for outlet_ind, geo in enumerate(char_val_dict["name"]):

        flow_list = char_val_dict["flow_list"][outlet_ind]
        dP_list = char_val_dict["dP_list"][outlet_ind]

        Q = np.asarray(flow_list).reshape(-1,1)


        r2_unsteady = 0; r2_steady = 0; r2_L = 0

        dP = np.asarray(dP_list).reshape(-1,1)
        X = np.hstack([np.square(Q), Q])
        coefs, residuals, t, q = np.linalg.lstsq(X, dP, rcond=None);
        r2_steady = get_r2(X, dP, coefs.reshape(-1,1))
        err = np.linalg.norm(residuals)/(1333**2)
        print(f"geo: {geo} {(np.linalg.norm(residuals)/(1333**2))}")
        if 1 < err <3:
            plt.plot(flow_list, dP_list)
            print("plotting")

        a = coefs[0][0]; b = coefs[1][0]
        #print(a,b)
        # if a < -75:
        #     print(geo)
        #     print(f"a:{a}")
        #     print(f"dP:{dP}")

        char_val_dict["coef_a"].append(a); char_val_dict["coef_b"].append(b)

        if unsteady:
            Q_unsteady = char_val_dict["unsteady_flow_list"][outlet_ind].reshape(-1,1)
            dQdt = char_val_dict["unsteady_flow_der_list"][outlet_ind].reshape(-1,1)

            dP_unsteady_total = char_val_dict["unsteady_dP_list"][outlet_ind].reshape(-1,1)
            dP_unsteady_component = char_val_dict["unsteady_dP_list"][outlet_ind].reshape(-1,1) - (a * np.square(Q_unsteady) + b * Q_unsteady)


            coefs, residuals, t, q = np.linalg.lstsq(dQdt, dP_unsteady_component, rcond=None);
            r2_unsteady = get_r2(dQdt, dP_unsteady_component, coefs.reshape(-1,1))
            L = coefs[0][0]
            char_val_dict["coef_L"].append(L)

            X_unsteady = np.hstack([np.square(Q_unsteady), Q_unsteady, dQdt])

            coefs, residuals, t, q = np.linalg.lstsq(X_unsteady, dP_unsteady_total, rcond=None);
            r2_UO = get_r2(X_unsteady, dP_unsteady_total, coefs.reshape(-1,1))
            a_UO = coefs[0][0]; b_UO = coefs[1][0]; L_UO =  coefs[2][0]
            char_val_dict["coef_a_UO"].append(a_UO); char_val_dict["coef_b_UO"].append(b_UO); char_val_dict["coef_L_UO"].append(L_UO)

        if err > 3:#r2_steady < 0.90 and r2_unsteady < 0.90 and r2_UO < 0.90:
            to_rm.append(outlet_ind)
            print(f"{geo} Steady R2: {r2_steady}.  Unsteady R2: {r2_unsteady}.")

    if rm_low_r2:
        print(f"Removing {len(to_rm)} outlets for low r2 values.")
        to_keep = []
        for i in range(int(len(char_val_dict["name"])/2)):
            if (2*i in to_rm) or 2*i+1 in to_rm:
                continue
            else:
                to_keep.append(2*i); to_keep.append(2*i + 1)

        for key in char_val_dict:
            try:
                char_val_dict[key] = [char_val_dict[key][ind] for ind in to_keep]
            except:
                continue
    plt.savefig("results/flow_vs_dp")
    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    return

def remove_outlier_coefs(anatomy, sd_tol):
    char_val_dict = load_dict(f"/home/nrubio/Desktop/aorta_synthetic_data_dict_{anatomy}")

    outlier_inds, non_outlier_inds = get_outlier_inds(char_val_dict["coef_a"][::2], m = sd_tol)

    non_outlier_inds = [2*ind for ind in non_outlier_inds] + [2*ind+1 for ind in non_outlier_inds]
    non_outlier_inds.sort()

    for key in char_val_dict.keys():
        try:
            full_array = char_val_dict[key]
            char_val_dict[key] = list(np.asarray(char_val_dict[key])[2*np.asarray(non_outlier_inds).astype(int)])
        except:
            continue

    print(f"a outlier_inds: {outlier_inds}")
    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    return

def get_geo_scalings(anatomy, unsteady = False):

    #plt.style.use('dark_background')

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict")
    scaling_dict = {}
    to_normalize = ["outlet_radius","inlet_radius","outlet_area","inlet_area", "angle", "coef_a", "coef_b", "inlet_length", "outlet_length"]
    if unsteady:
        to_normalize.append("coef_L")
        to_normalize.append("coef_a_UO")
        to_normalize.append("coef_b_UO")
        to_normalize.append("coef_L_UO")
    for value in to_normalize:

        scaling_dict.update({value: [np.mean(char_val_dict[value]), np.std(char_val_dict[value])]})

        plt.clf()
        plt.hist(char_val_dict[value], bins = 30, alpha = 0.5, label = "outlet1")
        plt.hist(char_val_dict[value], bins = 30, alpha = 0.5, label = "outlet2")
        plt.xlabel(value); plt.ylabel("frequency"); plt.title("Synthetic Aorta Data");
        plt.legend()
        plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_{value}_both.png", bbox_inches='tight', transparent=False, format = "png")

        plt.clf()
        plt.scatter(char_val_dict[value], char_val_dict["coef_a"])
        plt.xlabel(value); plt.ylabel("a"); plt.title(f"Synthetic {anatomy} Data");
        plt.savefig(f"results/synthetic_data_trends/a_trends/{anatomy}_{value}.png", bbox_inches='tight', transparent=False, format = "png")

        plt.clf()
        plt.scatter(char_val_dict[value], np.asarray(char_val_dict["coef_b"]))
        plt.xlabel(value); plt.ylabel("b"); plt.title(f"Synthetic {anatomy} Data");
        plt.savefig(f"results/synthetic_data_trends/b_trends/{anatomy}_{value}.png", bbox_inches='tight', transparent=False, format = "png")

        if unsteady:
            plt.clf()
            plt.scatter(char_val_dict[value], np.asarray(char_val_dict["coef_L"]))
            plt.xlabel(value); plt.ylabel("L"); plt.title(f"Synthetic {anatomy} Data");
            plt.savefig(f"results/synthetic_data_trends/L_trends/{anatomy}_{value}.png", bbox_inches='tight', transparent=False, format = "png")

    plt.clf()
    plt.hist([dP_list_ind[-1] for dP_list_ind in char_val_dict["dP_list"]], bins = 30, alpha = 0.5, label = "outlet1")
    plt.xlabel(value); plt.ylabel("frequency"); plt.title(f"Synthetic {anatomy} Data");
    plt.legend()
    plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_dP_both.png", bbox_inches='tight', transparent=False, format = "png")

    plt.clf()
    plt.hist([flow_list_ind[-1] for flow_list_ind in char_val_dict["flow_list"]], bins = 30, alpha = 0.5, label = "outlet1")
    plt.xlabel(value); plt.ylabel("frequency"); plt.title(f"Synthetic {anatomy} Data");
    plt.legend()
    plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_flow_both.png", bbox_inches='tight', transparent=False, format = "png")

    plt.clf()
    plt.scatter(char_val_dict["outlet_radius"], [dP_list_ind[-1] for dP_list_ind in char_val_dict["dP_list"]])
    plt.xlabel("outlet_radius"); plt.ylabel("pressure_differential"); plt.title(f"Synthetic {anatomy} Data");
    plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_radius_dp.png", bbox_inches='tight', transparent=False, format = "png")

    plt.clf()
    plt.scatter([flow_list_ind[-1] for flow_list_ind in char_val_dict["flow_list"]], [dP_list_ind[-1] for dP_list_ind in char_val_dict["dP_list"]])
    plt.xlabel("flow"); plt.ylabel("pressure_differential"); plt.title(f"Synthetic {anatomy} Data");
    plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_flow_dp.png", bbox_inches='tight', transparent=False, format = "png")


    save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_scaling_dict")
    return
