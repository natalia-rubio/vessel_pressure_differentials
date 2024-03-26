import os
import vtk
import argparse
import pdb
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pickle

class Integration:
    """
    Class to perform integration on slices
    """

    def __init__(self, inp):
        try:
            self.integrator = vtk.vtkIntegrateAttributes()
        except AttributeError:
            raise Exception('vtkIntegrateAttributes is currently only supported by pvpython')

        if not inp.GetOutput().GetNumberOfPoints():
            raise Exception('Empty slice')

        self.integrator.SetInputData(inp.GetOutput())
        self.integrator.Update()

    def evaluate(self, res_name):
        """
        Evaluate integral.
        Distinguishes between scalar integration (e.g. pressure) and normal projection (velocity)
        Optionally divides integral by integrated area
        Args:
            field: pressure, velocity, ...
            res_name: name of array
        Returns:
            Scalar integral
        """
        # type of result
        field = res_name.split('_')[0]

        if field == 'velocity':
            int_name = 'normal_' + res_name
        else:
            int_name = res_name

        # evaluate integral
        integral = v2n(self.integrator.GetOutput().GetPointData().GetArray(int_name))[0]

        # choose if integral should be divided by area
        if field == 'velocity':
            return integral
        else:
            return integral / self.area()

    def area(self):
        """
        Evaluate integrated surface area
        Returns:
        Area
        """
        return v2n(self.integrator.GetOutput().GetCellData().GetArray('Area'))[0]



def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res


def get_all_arrays(geo):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())

    return point_data, cell_data

def split(array):
    """
    Split array name in name and time step if possible
    """
    comp = array.split('_')
    num = comp[-1]

    # check if array name has a time step
    try:
        time = float(num)
        name = '_'.join([c for c in comp[:-1]])
    except ValueError:
        time = 0
        name = array
    return time, name


def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh
    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def write_geo(fname, input):
    """
    Write geometry to file
    Args:
        fname: file name
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def threshold(inp, t, name):
    """
    Threshold according to cell array
    Args:
        inp: InputConnection
        t: BC_FaceID
        name: name in cell data used for thresholding
    Returns:
        reader, point data
    """
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(inp)
    thresh.SetInputArrayToProcess(0, 0, 0, 1, name)
    thresh.ThresholdBetween(t, t)
    thresh.Update()
    return thresh


def calculator(inp, function, inp_arrays, out_array):
    """
    Function to add vtk calculator
    Args:
        inp: InputConnection
        function: string with function expression
        inp_arrays: list of input point data arrays
        out_array: name of output array
    Returns:
        calc: calculator object
    """
    calc = vtk.vtkArrayCalculator()
    for a in inp_arrays:
        calc.AddVectorArrayName(a)
    calc.SetInputData(inp.GetOutput())
    if hasattr(calc, 'SetAttributeModeToUsePointData'):
        calc.SetAttributeModeToUsePointData()
    else:
        calc.SetAttributeTypeToPointData()
    calc.SetFunction(function)
    calc.SetResultArrayName(out_array)
    calc.Update()
    return calc


def cut_plane(inp, origin, normal):
    """
    Cuts geometry at a plane
    Args:
        inp: InputConnection
        origin: cutting plane origin
        normal: cutting plane normal
    Returns:
        cut: cutter object
    """
    # define cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    # define cutter
    cut = vtk.vtkCutter()
    cut.SetInputData(inp)
    cut.SetCutFunction(plane)
    cut.Update()
    return cut


def get_points_cells(inp):
    cells = []
    for i in range(inp.GetOutput().GetNumberOfCells()):
        cell_points = []
        for j in range(inp.GetOutput().GetCell(i).GetNumberOfPoints()):
            cell_points += [inp.GetOutput().GetCell(i).GetPointId(j)]
        cells += [cell_points]
    return v2n(inp.GetOutput().GetPoints().GetData()), np.array(cells)


def connectivity(inp, origin):
    """
    If there are more than one unconnected geometries, extract the closest one
    Args:
        inp: InputConnection
        origin: region closest to this point will be extracted
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp.GetOutput())
    con.SetExtractionModeToClosestPointRegion()
    con.SetClosestPoint(origin[0], origin[1], origin[2])
    con.Update()
    return con


def connectivity_all(inp):
    """
    Color regions according to connectivity
    Args:
        inp: InputConnection
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp)
    con.SetExtractionModeToAllRegions()
    con.ColorRegionsOn()
    con.Update()
    assert con.GetNumberOfExtractedRegions() > 0, 'empty geometry'
    return con


def extract_surface(inp):
    """
    Extract surface from 3D geometry
    Args:
        inp: InputConnection
    Returns:
        extr: vtkExtractSurface object
    """
    extr = vtk.vtkDataSetSurfaceFilter()
    extr.SetInputData(inp)
    extr.Update()
    return extr.GetOutput()


def clean(inp):
    """
    Merge duplicate Points
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(inp)
    # cleaner.SetTolerance(1.0e-3)
    cleaner.PointMergingOn()
    cleaner.Update()
    return cleaner.GetOutput()


def scalar_array(length, name, fill):
    """
    Create vtkIdTypeArray array with given name and constant value
    """
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfValues(length)
    ids.SetName(name)
    ids.Fill(fill)
    return ids


def add_scalars(inp, name, fill):
    """
    Add constant value array to point and cell data
    """
    inp.GetOutput().GetCellData().AddArray(scalar_array(inp.GetOutput().GetNumberOfCells(), name, fill))
    inp.GetOutput().GetPointData().AddArray(scalar_array(inp.GetOutput().GetNumberOfPoints(), name, fill))


def rename(inp, old, new):
    if inp.GetOutput().GetCellData().HasArray(new):
        inp.GetOutput().GetCellData().RemoveArray(new)
    if inp.GetOutput().GetPointData().HasArray(new):
        inp.GetOutput().GetPointData().RemoveArray(new)
    inp.GetOutput().GetCellData().GetArray(old).SetName(new)
    inp.GetOutput().GetPointData().GetArray(old).SetName(new)


def replace(inp, name, array):
    arr = n2v(array)
    arr.SetName(name)
    inp.GetOutput().GetCellData().RemoveArray(name)
    inp.GetOutput().GetCellData().AddArray(arr)


def geo(inp):
    poly = vtk.vtkGeometryFilter()
    poly.SetInputData(inp)
    poly.Update()
    return poly.GetOutput()

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def get_length(locs):
    length = 0
    for i in range(1, locs.shape[0]):
        length += np.linalg.norm(locs[i, :] - locs[i-1, :])
    return length


def get_res_names(inp, res_fields):
    # result name list
    res = []

    # get integral for each result
    for i in range(inp.GetPointData().GetNumberOfArrays()):
        res_name = inp.GetPointData().GetArrayName(i)
        field = res_name.split('_')[0]
        num = res_name.split('_')[-1]

        # check if field should be added to output
        if field in res_fields:
            try:
                float(num)
                res += [res_name]
            except ValueError:
                pass

    return res