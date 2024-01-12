import os
import sys
import math
import numpy as np
import sv
import pickle
from sv import *
import vtk
import os
import platform
from vtk.util import numpy_support

# LOAD IN CONTOUR GROUPS AND PATHLINES
# HERE

#####################################
# FUNCTIONS FOR LOFT GENERATION
#####################################

options = geometry.LoftNurbsOptions()

def robust_union(model_1,model_2):
    """
    Union two capped SV solid objects into one sv solid object.

    PARAMETERS:
    model_1: (sv.modeling.solid): first solid model
    model_2: (sv.modeling.solid): second solid model
    """
    modeler = modeling.Modeler(modeling.Kernel.POLYDATA)
    model_1_be = bad_edges(model_1)
    model_2_be = bad_edges(model_2)
    print("Model 1 Bad Edges: {}\n Model 2 Bad Edges: {}".format(model_1_be,model_2_be))
    if model_1_be == 0 and model_2_be == 0:
        print("starting union")
        unioned_model = modeler.union(model_1,model_2)
        print("finished union")
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        print("intermediate union completed")
        if bad_edges(unioned_model) > 0:
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            print('Filling')
            unioned_model = fill(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            print('Cleaning')
            unioned_model = clean(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            unioned_model = tri(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
        print('union successful')
        return unioned_model
    else:
        print('1 or both models have bad edges.')
        raise NameError('Bad Edges.  Skipping this set of geometric parameters.')
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        return unioned_model

def union_all(solids,n_cells=100, max_density = 100):
    """
    Union a list of all vessels together.

    PARAMETERS:
    solids:   (list): list of capped sv solid objects

    RETURNS:
    joined  (sv.modeling.solid): sv solid object
    """
    if len(solids) == 1:
        print("Single body.  No unioning needed.")
        return solids[0]
    else:
        for i in range(len(solids)):
            solids[i] = norm(solids[i])
            solids[i] = remesh(solids[i]) #, cell_density_mm=[max_density,min(int(max_density/2), 10)])
            solids[i] = remesh_caps(solids[i])
            print("completed first remeshing (pre-union)")
        joined = robust_union(solids[0],solids[1])
        for i in range(2,len(solids)):
            print("UNION NUMBER: "+str(i)+"/"+str(len(solids)))
            joined = robust_union(joined,solids[i])
            if joined is None:
                print("unioning failed")
                return None
        print("unioning passed")
        return joined

def bad_edges(model):
    fe = vtk.vtkFeatureEdges()
    fe.FeatureEdgesOff()
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.SetInputData(model.get_polydata())
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()

def clean(model):
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.PointMergingOn()
    clean_filter.SetInputData(model.get_polydata())
    clean_filter.Update()
    model.set_surface(clean_filter.GetOutput())
    print("Model cleaned.")
    return model

def tri(model):
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(model.get_polydata())
    tri_filter.Update()
    model.set_surface(tri_filter.GetOutput())
    return model

def fill(model):
    poly = vmtk.cap(surface=model.get_polydata(),use_center=False)
    model.set_surface(poly)
    return model

def surf_area(poly):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return mass.GetSurfaceArea()

def remesh(model,cell_density_mm=[100,10]):
    """
    PARAMTERS:
    model:        SV solid modeling object
    cell density: number of mesh elements per square
                  mm. Given as an acceptable range.
    """
    poly = model.get_polydata()
    poly_sa = surf_area(poly)*100
    cell_num_hmin = max(5,int(cell_density_mm[0]*poly_sa))
    cell_num_hmax = min(5,int(cell_density_mm[1]*poly_sa)) # CHANGED TO MIN
    hmin = (((poly_sa/100)/cell_num_hmin)*2)**(1/2)
    hmax = (((poly_sa/10)/cell_num_hmax)*2)**(1/2)
    print("Remeshing Model:\nhmin: ----> {}\nhmax ----> {}".format(hmin,hmax))

    remeshed_polydata = mesh_utils.remesh(model.get_polydata(),hmin=hmin,hmax=hmax)
    model.set_surface(remeshed_polydata)
    return model

def remesh_face(model,face_id,cell_density_mm=40):
    face_poly = model.get_face_polydata(face_id)
    face_sa = surf_area(face_poly)*100
    cell_num = max(5,int(cell_density_mm*face_sa))
    edge_size = round((((face_sa/100)/cell_num)*2)**(1/2),5)
    edge_size = max(0.01,edge_size)
    print("Remeshing Face: {} ----> Edge Size: {}".format(face_id,edge_size))
    remeshed_poly = mesh_utils.remesh_faces(model.get_polydata(),[face_id],edge_size)
    model.set_surface(remeshed_poly)
    return model

def remesh_caps(model,cell_density_mm=40):
    cap_ids = model.identify_caps()
    face_ids = model.get_face_ids()
    for i,c in enumerate(cap_ids):
        if c:
            model = remesh_face(model,face_ids[i],cell_density_mm=cell_density_mm)
    return model

def norm(model):
    """
    Determine the normal vectors along the
    polydata surface.

    PARAMETERS
    model:    SV solid modeling object
    """
    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.AutoOrientNormalsOn()
    norm_filter.ComputeCellNormalsOn()
    #norm_filter.FlipNormalsOn()
    norm_filter.ConsistencyOn()
    norm_filter.SplittingOn()
    norm_filter.NonManifoldTraversalOn()
    norm_filter.SetInputData(model.get_polydata())
    norm_filter.Update()
    model.set_surface(norm_filter.GetOutput())
    print("Model norms set.")
    return model

def loft(contours,num_pts=40,distance=False):
    """
    Generate an open lofted NURBS surface along a given
    vessel contour group.

    PARAMETERS:
    contours (list):  list of contour polydata objects defining one vessel.
    num_pts  (int) :  number of sample points to take along each contour.
    distance (bool):  flag to use distance based method for contour alignment
    """
    print("contours")
    print(contours)
    for idx in range(len(contours)):
        contours[idx] = geometry.interpolate_closed_curve(polydata=contours[idx],number_of_points=num_pts)

        print("index")
        print(idx)
        if idx != 0:
            contours[idx] = geometry.align_profile(contours[idx-1],contours[idx],distance)
    options = geometry.LoftNurbsOptions()
    loft_polydata = geometry.loft_nurbs(polydata_list=contours,loft_options=options)
    loft_solid = modeling.PolyData()
    loft_solid.set_surface(surface=loft_polydata)
    return loft_solid

def loft_all(contour_list):
    """
    Loft all vessels defining the total model that you want to create.

    PARAMETERS
    contour_list: (list): list of lists that contain polydata contour groups
                          Example for two vessels:

                          contour_list -> [[polydataContourObject1,polydataContourObject2],[polydataContourObject1,polydataContourObject2]]

    RETURNS:
    lofts:        (list): list of open sv solid models of the lofted 3D surface. Note that
                          the loft is not yet capped.
    """
    lofts = []
    for group in contour_list:
        lofts.append(loft(group))

    #lofts.append(loft(contour_list))
    return lofts

def cap_all(loft_list):
    """
    Cap all lofted vessels.

    PARAMETERS:
    loft_list  (list): list of sv modeling solid objects that are open lofts generated from
                       the 'loft_all' function.

    RETURNS:
    capped     (list): list of capped solids
    """
    capped = []
    for loft_solid in loft_list:
        capped_solid = modeling.PolyData()
        capped_solid.set_surface(vmtk.cap(surface=loft_solid.get_polydata(),use_center=False))
        capped_solid.compute_boundary_faces(angle=85)
        capped.append(capped_solid)
    return capped

def check_cap_solids(cap_solid_list):
    """
    Check capped solids for bad edges.
    """
    for solid in cap_solid_list:
        if bad_edges(solid) > 0:
            return False
    return True

def create_vessels(contour_list,attempts=3):
    """
    create seperate capped vessels for all contour groups defining a model of interest.

    PARAMETERS:
    contour_list: (list): list of lists of contour polydata objects defining individual vessels
                          within the total model.
    attemps:      (int) : the number of times that bad edges correction will be attemped during loft
                          alignment
    """
    print("Creating vessels.")
    i = 0
    success = False
    while not success and i < attempts:
        print("attempt " + str(i))
        lofts = loft_all(contour_list)
        cap_solids = cap_all(lofts)
        success = check_cap_solids(cap_solids)
        if success:
            print('Lofting Passed')
        else:
            print('Lofting Failed')
            i = i +1

        if i == attempts:
            raise NameError('Unable to loft.  Skipping this set of geometric parameters.')

    return cap_solids

def combine_walls(model):
    print("Combining walls")
    caps = model.identify_caps()
    ids = model.get_face_ids()
    print("ids " + str(ids))
    walls = [ids[i] for i,x in enumerate(caps) if not x]

    while len(walls) > 1:
        print("walls " + str(walls))
        target = walls[0]
        lose = [walls[1]]
        model.combine_faces(target,lose)
        #model.set_surface(model)
        ids = model.get_face_ids()
        caps = model.identify_caps()
        walls = [ids[i] for i,x in enumerate(caps) if not x]
        print(walls)
    print("Walls combined.")

    return model, walls, caps, ids

def combine_caps(model, walls, ids, num_caps = 3):
    print("Combining caps.")
    print(len(ids))
    if len(ids) > num_caps+len(walls):
        face_cells = []
        for idx in ids:
            face = model.get_face_polydata(idx)
            cells = face.GetNumberOfCells()
            print(cells)
            face_cells.append(cells)
        data_to_remove = len(ids) - num_caps
        print(ids)
        print(data_to_remove)
        remove_list = []
        for i in range(data_to_remove):
            remove_list.append(ids[face_cells.index(min(face_cells))])
            face_cells[face_cells.index(min(face_cells))] += 1000
        print(remove_list)
        while len(remove_list) > 0:
            target = walls[0]
            lose = remove_list.pop(-1)
            try:
                print("trying to combine cap face")
                print(lose)
                model.combine_faces(target,[lose])
            except:
                print("Failed. Target: ")
                print(target)
                print("/nLose: ")
                print(lose)
            #model.set_surface(combined)
            print(remove_list)
        print(model.get_face_ids())
    print("Caps combined")
    return model

def get_smoothed_model(model, smoothing_radius):
    smooth_model = model.get_polydata()

    smoothing_params = {'method':'constrained', 'num_iterations':5}

    smooth_model = sv.geometry.local_sphere_smooth(surface = smooth_model, radius = smoothing_radius, center = [0, 0, 0], smoothing_parameters = smoothing_params)

    return smooth_model

def get_max_area_cap(mesher, walls):
    print("finding max area cap")
    areas = []
    mass = vtk.vtkMassProperties()
    mesh = modeling.PolyData()
    faces = mesher.get_model_face_ids()
    print("got model face ids")
    assert walls[0] == faces[0], "first face is wall"
    for face in faces:
        if face == walls[0]:
            continue

        mass.SetInputData(mesher.get_face_polydata(face))
        areas.append(mass.GetSurfaceArea())
        print("areas: "); print(areas)
    ind = np.argmax(np.asarray(areas)) + 1
    max_area_cap = faces[ind]
    print("faces" + str(faces))
    print("max_area_cap" + str(max_area_cap))
    return max_area_cap

def get_inlet_cap(mesher, walls):
    print("finding max area cap")
    distance_to_origin = []
    mass = vtk.vtkMassProperties()
    mesh = modeling.PolyData()
    faces = mesher.get_model_face_ids()
    print("got model face ids")
    assert walls[0] == faces[0], "first face is wall"
    for face in faces:
        if face == walls[0]:
            continue
        pts = numpy_support.vtk_to_numpy(mesher.get_face_polydata(face).GetPoints().GetData())
        x_loc = np.mean(pts[:,0])
        y_loc = np.mean(pts[:,1])
        distance_to_origin.append(np.sqrt(x_loc**2 + y_loc**2))

    ind = np.argmin(np.asarray(distance_to_origin)) + 1
    max_area_cap = faces[ind]
    print("faces" + str(faces))
    print("max_area_cap" + str(max_area_cap))
    return max_area_cap
