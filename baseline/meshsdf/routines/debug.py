import torch_geometric
from transforms import SignedDistanceField
from datasets import InMemoryVesselDataset
import numpy as np


def run():

    transforms = [
        torch_geometric.transforms.GenerateMeshNormals(),
        SignedDistanceField(split=True)
    ]

    dataset = InMemoryVesselDataset(root="/home/sukjm/data/cell-datasets/cell00702_vtk/",
                                    pattern="cell_00702_man_seg*.vtk",
                                    split=[0, 1],
                                    purpose="debug",
                                    pre_transform=torch_geometric.transforms.Compose(transforms))

    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    for sample in dataloader:
        np.savez("data/cell/samples/cell", pos=sample.positive.numpy(), neg=sample.negative.numpy())
