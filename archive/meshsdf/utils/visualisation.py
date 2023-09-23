import vtk
from utils.vtk_tools import torch_to_vtk, add_fields


def append_file(sample, filename, fields=None):

    # Read the mesh data
    path = sample.dir[0]
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()

    # Attach an arbitrary number of fields
    polydata = add_fields(polydata, fields)

    # Save the VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Update()


def new_file(points, polygons, filename, fields=None):

    # Create the "vtkPolyData" object
    polydata = torch_to_vtk(points, polygons, fields)

    # Write the files
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
