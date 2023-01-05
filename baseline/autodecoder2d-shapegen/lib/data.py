#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio

from lib.utils import *


class InMemorySDFSamplesFraction(torch.utils.data.Dataset):
    def __init__(
        self,
        fraction = 1.0,
        no_rotations = 10 # generate this many rotations
    ):
        self.fraction = fraction
        self.no_rotations = no_rotations    
        
        #self.sample_size = int(self.fraction * len(self.dataset[0][0]))    

        # Load the whole dataset into CPU memory
        #self.dataset = []
        #print("loading dataset into memory")
        #for idx, matfile in enumerate(self.matfiles):
        #    filename = os.path.join(self.data_source, matfile)
        #    self.dataset.extend(unpack_sdf_samples_fraction(filename, self.fraction, sequence_id=idx))
            
        # Load the whole dataset into CPU memory
        self.dataset = []
        rotations_deg = np.linspace(0, 180, num=no_rotations, endpoint=False)
        print("Generating and loading dataset into memory")
        for idx in range(no_rotations):
            #filename = os.path.join(self.data_source, self.matfiles[idx])
            self.dataset.append(
                (unpack_sdf_samples_fraction('triangle', self.fraction,
                                             sdf_id=idx, 
                                             angle_deg=rotations_deg[idx]), idx)
            )

    def __len__(self): # total number of training shapes
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]
        

'''class InMemorySDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.matfiles = get_instance_filenames(data_source)

        # Load the whole dataset into CPU memory
        self.dataset = []
        print("loading dataset into memory (takes approx. 2:00 [min] for 50 samples)")
        for idx in range(len(self.matfiles)):
            filename = os.path.join(self.data_source, self.matfiles[idx])
            self.dataset.append(
                (unpack_sdf_samples(filename, self.subsample), idx, self.matfiles[idx])
            )

    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, idx):
        return self.dataset[idx]'''


'''class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.matfiles = get_instance_filenames(data_source)

    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.matfiles[idx]
        )
        return unpack_sdf_samples(filename, self.subsample), idx, self.matfiles[idx]'''


'''class RGBA2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, RGBA, intrinsic, extrinsic, mesh_name'''
