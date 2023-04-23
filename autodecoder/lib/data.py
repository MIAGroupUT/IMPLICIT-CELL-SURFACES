#!/usr/bin/env python3

import numpy as np
import os
import torch
import torch.utils.data
from pytorch3d.transforms import euler_angles_to_matrix # 3D rotation matrix

from lib.utils import *


class InMemorySDFSamplesFraction(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        fraction = 1.0
    ):
        self.fraction = fraction
        self.data_source = data_source
        self.matfiles = get_instance_filenames(data_source)    

        # Load the whole dataset into CPU memory
        self.dataset = []
        print("loading dataset into memory")
        for idx, matfile in enumerate(self.matfiles):
            filename = os.path.join(self.data_source, matfile)
            self.dataset.extend(unpack_sdf_samples_fraction(filename, self.fraction, sequence_id=idx))

    def __len__(self): # total number of frames across all sequences
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def unpack_sdf_samples_fraction(filename, fraction, sequence_id=0):

    print('Processing', filename)
    
    # load matlab file
    sdf_t = np.asarray([sio.loadmat(filename)['sdf_vid']], dtype=np.float)
    sdf_t = np.squeeze(sdf_t)

    frames = []  # or "scenes"
    for t, sdf in enumerate(tqdm(sdf_t, desc="sequence {}".format(sequence_id), leave=False)):
    
        # create coordinate grid
        pixel_coords = np.stack(np.mgrid[:sdf.shape[0], :sdf.shape[1],
                                         :sdf.shape[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sdf.shape[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sdf.shape[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sdf.shape[2] - 1)
        pixel_coords = np.squeeze(pixel_coords)
        
        # normalize from [0, 1] to [-1, 1]
        pixel_coords -= 0.5
        pixel_coords *= 2.0
        
        # flatten all but the last dimension
        pixel_coords = pixel_coords.reshape(-1, pixel_coords.shape[-1])

        # combine grid & corresponding SDF values
        samples = torch.zeros(sdf.shape[0]*sdf.shape[1]*sdf.shape[2], 4)

        sdf = sdf.flatten()
        samples[:, 0:3] = torch.from_numpy(pixel_coords).float()
        samples[:, 3] = torch.from_numpy(sdf).float()
    
        sample_size = int(samples.shape[0] * fraction)
        
        # separate inner and outer points
        index_ip = np.where(samples[:,3] <= 0.1)
        ip_tensor = samples[index_ip,:]
        index_op = np.where(samples[:,3] > 0.1)
        op_tensor = samples[index_op,:]  
        
        # 70% of samples are inner points
        no_ip = int(sample_size * 0.7)
        if len(index_ip[0]) < no_ip:
            no_ip = len(index_ip[0])
            samples_ip = ip_tensor
        else:
            ip_rnd_idx = torch.randint(0, ip_tensor.shape[1], (no_ip,))
            samples_ip = torch.index_select(ip_tensor, 1, ip_rnd_idx)
            
        no_op = sample_size - no_ip
        
        op_rnd_idx = torch.randint(0, op_tensor.shape[1], (no_op,))      
        samples_op = torch.index_select(op_tensor, 1, op_rnd_idx)
    
        samples = torch.cat([samples_ip, samples_op], 1).squeeze().float()
        
        frames.append((samples, t, filename, sequence_id))
   
    # time coordinates
    t = np.asarray([frame[1] for frame in frames], dtype='float32')
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    
    frames = [(scene[0], t[i], *scene[2:]) for i, scene in enumerate(frames)] 

    return frames
            

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


def create_SDF(decoder, latent_vec, time, rot_ang, N=np.array([128,128,128]),
               max_batch=64**3, offset=None, scale=None):

    # pre-allocate array
    samples = torch.zeros(N[0]*N[1]*N[2], 5)
    
    # prepare coordinate grid
    pixel_coords = np.stack(np.mgrid[:N[0], :N[1], :N[2]], 
                            axis=-1)[None, ...].astype(np.float32)
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(N[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / (N[1] - 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / (N[2] - 1)
    pixel_coords = np.squeeze(pixel_coords)
    
    # normalize from [0, 1] to [-1, 1]
    pixel_coords -= 0.5
    pixel_coords *= 2.  
    
    # flatten all but the last dimension
    pixel_coords = pixel_coords.reshape(-1, pixel_coords.shape[-1])
    
    # rotate cootdinates
    pixel_coords = torch.from_numpy(pixel_coords) @ \
        euler_angles_to_matrix(torch.from_numpy(rot_ang), 'XYZ')
    
    # paste grid into the samples array
    samples[:, 0:3] = pixel_coords
    samples[:, 3] = time.expand(samples.size(0))
    
    samples.requires_grad = False

    num_samples = N[0]*N[1]*N[2] 

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:4].cuda()
        samples[head : min(head + max_batch, num_samples), 4] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch
    
    # reshape SDF
    sdf_values = samples[:, 4]
    sdf_values = sdf_values.reshape(N[0], N[1], N[2])

    return sdf_values.detach().cpu().numpy().astype(np.float32)
