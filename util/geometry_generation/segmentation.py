import path_planning
import radius_planning
import pdb
import sv


def get_vessel_segmentations(geo_params):
    """
    """
    num_pts = 10

    # Get path
    path_list = path_planning.get_path(length=geo_params["length"], 
                                  curvature=geo_params["curvature"], 
                                  elem_length=geo_params["length"]/num_pts)
    
    radii_list = radius_planning.get_radii(length=geo_params["length"],
                                    inlet_radius=geo_params["inlet_radius"],
                                    outlet_radius=geo_params["outlet1_radius"],
                                    stenosis_dict=geo_params["stenosis_dict"],
                                    elem_length=geo_params["length"]/num_pts)

    path = sv.pathplanning.Path()
    for point in path_list:
        path.add_control_point(point)
    print(path_list)

    assert path_list == path.get_control_points()
    assert len(path_list) == len(radii_list)
    path_curve_points = path.get_curve_points()


    segmentations = []
    for i in range(len(path_list)):
        contour = sv.segmentation.Circle(radius = radii_list[i],
                                    center = path_list[i],
                                    normal = path.get_curve_tangent(path_curve_points.index(path_list[i])))
        segmentations.append(contour)
    return path, segmentations

if __name__ == '__main__':
    geo_params = {"length": 10, 
                "curvature": 90,
                "inlet_radius": 1,
                "outlet1_radius": 2,
                "outlet2_radius": 2,
                "stenosis_dict": {"magnitude": 0.5, 
                                "spread": 0.1,
                                "location": 0.5}}
    path, segmentations = get_vessel_segmentations(geo_params)
    pdb.set_trace()
    # plot_segmentations(segmentations)
    # pdb.set_trace()
# to run with sv python --> /Applications/SimVascular.app/Contents/Resources/simvascular --python --
