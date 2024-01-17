import sys
import vtk
import os
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm

from util.get_bc_integrals import get_res_names
#from util.junction_proc import *
from geo_processing import *
from util.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle
from sklearn.linear_model import LinearRegression

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def get_length(locs):
    length = 0
    for i in range(1, locs.shape[0]):
        length += np.linalg.norm(locs[i, :] - locs[i-1, :])
    return length


def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # cut 3d geometry
    cut_3d = cut_plane(inp_3d, origin, normal)
    #write_geo(f'slice_{origin[0]}.vtp', cut_3d.GetOutput())

    # extract region closest to centerline
    con = connectivity(cut_3d, origin)
    #write_geo(f'con_{origin[0]}.vtp', con.GetOutput())
    return con


def get_integral(inp_3d, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # slice vessel at given location
    inp = slice_vessel(inp_3d, origin, normal)

    # recursively add calculators for normal velocities
    for v in get_res_names(inp_3d, 'velocity'):
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        #fun = 'dot(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+',' + v + ")"
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)

def extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 1000):

    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()# get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])# get point and normals from centerline
    points = v2n(reader_1d.GetPoints().GetData())
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))# initialize output

    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array) # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3 # integrate results on all points of intergration cells
    print(f"Extracting solution at {reader_1d.GetNumberOfPoints()} points.")
    for i in tqdm(range(reader_1d.GetNumberOfPoints())):
        # check if point is cap
        reader_1d.GetPointCells(i, ids)
        if ids.GetNumberOfIds() == 1:
            if gid[i] == 0:
                # inlet
                points[i] += eps_norm * normals[i]
            else:
                # outlets
                points[i] -= eps_norm * normals[i]
        else:
            if only_caps:
                continue # create integration object (slice geometry at point/normal)

        try:
            #import pdb; pdb.set_trace()
            integral = get_integral(reader_3d, points[i], normals[i])
        except Exception:
            continue # integrate all output arrays

        for name in res_names:
            reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    write_geo(fpath_out, reader_1d)
    return

def get_soln_dict(sol_path1d, fpath_out, only_caps=False, num_time_steps = 1000):
    """
    Get all result array names
    Args:
        inp: vtk InputConnection
        names: list of names to search for
    Returns:
        list of result array names
    """
    sol1d = read_geo(fpath_1d).GetOutput()
    return res_names