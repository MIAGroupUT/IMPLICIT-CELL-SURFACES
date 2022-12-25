import trimesh
import mesh_to_sdf
import numpy as np
import torch


class SignedDistanceField(object):
    """Computes a signed distance field.

    Args:
        split (bool): Whether to split positive and negative SDF values.
    """

    def __init__(self, split=False):
        self.split = split

    def __call__(self, data):
        mesh = trimesh.Trimesh(vertices=data.pos.numpy(), faces=data.face.t().numpy())
        mesh = mesh_to_sdf.utils.scale_to_unit_sphere(mesh)

        resolution = [np.linspace(-1., 1., num=32)] * 3
        x, y, z = np.meshgrid(*resolution)
        grid = np.vstack([x.ravel(), y.ravel(), z.ravel()]).transpose()

        sdf = mesh_to_sdf.mesh_to_sdf(mesh, grid)

        if self.split:
            index = np.where(sdf >= 0.)
            data.positive = torch.from_numpy(np.hstack([grid[index], sdf[index][..., None]]))
            index = np.where(sdf < 0.)
            data.negative = torch.from_numpy(np.hstack([grid[index], sdf[index][..., None]]))

        else:
            data.grid = torch.from_numpy(grid)
            data.sdf = torch.from_numpy(sdf)

        return data

    def __repr__(self):
        return '{}(split={})'.format(self.__class__.__name__, self.split)
