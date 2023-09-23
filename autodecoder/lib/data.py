#!/usr/bin/env python3

import numpy as np
import os
import torch
import torch.utils.data
from pytorch3d.transforms import euler_angles_to_matrix # 3D rotation matrix

from lib.utils import *


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        fraction = 1.0,
        otf_sampling = False,
        use_hdf5 = False
    ):
        self.fraction = fraction
        self.otf_sampling = otf_sampling
        self.use_hdf5 = use_hdf5
        self.data_source = data_source
        self.datafiles = get_instance_filenames(data_source, self.use_hdf5)    

        self.dataset = []
        if not self.otf_sampling:
            # load samples into CPU memory
            print("loading samples into CPU memory")
            for idx, datafile in enumerate(self.datafiles):
                filename = os.path.join(self.data_source, datafile)
                self.dataset.extend(unpack_sdf_samples(filename, idx,
                                                       self.fraction,
                                                       self.use_hdf5))           
        else:
            print("loading the whole dataset into CPU memory")
            for idx, datafile in enumerate(self.datafiles):
                filename = os.path.join(self.data_source, datafile)
                self.dataset.extend(unpack_sdf_samples(filename, idx, 1.0,
                                                       self.use_hdf5)) 
    
    def __len__(self): # total number of frames across all sequences
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.otf_sampling:
            # sample dataset on-the-fly (slower)
            return sample_fraction(self.dataset[idx], self.fraction)
        else:
            # get samples from CPU memory
            return self.dataset[idx]


def sample_fraction(data, fraction):
    
    # determine the number of SDF samples
    sample_size = int(data[0].shape[0] * fraction)
    
    # separate inner and outer points
    index_ip = torch.where(data[0][:,3] <= 0.3)
    ip_arr = torch.index_select(data[0], 0, index_ip[0])
    index_op = torch.where(data[0][:,3] > 0.3)
    op_arr = torch.index_select(data[0], 0, index_op[0])
    # free memory
    del index_ip, index_op
    
    # 70% of samples will be inner points
    no_ip = int(sample_size * 0.7)
    #if len(index_ip[0]) < no_ip:
    if ip_arr.shape[0] < no_ip:
        no_ip = len(index_ip[0])
        samples_ip = ip_arr
    else:
        ip_rnd_idx = torch.randint(0, ip_arr.shape[0], (no_ip,))
        samples_ip = torch.index_select(ip_arr, 0, ip_rnd_idx)
    # free memory
    del ip_arr, ip_rnd_idx

    # the rest will be outer points    
    no_op = sample_size - no_ip
    op_rnd_idx = torch.randint(0, op_arr.shape[0], (no_op,))      
    samples_op = torch.index_select(op_arr, 0, op_rnd_idx)   
    # free memory
    del op_arr, op_rnd_idx

    # concatenate randomly selected inner and outer points
    samples = torch.cat([samples_ip, samples_op], 0)
    
    return (samples, data[1], data[2], data[3])


def unpack_sdf_samples(filename, sequence_id, fraction=1.0, use_hdf5=False):

    print('Processing', filename)
    
    if use_hdf5:
        # load HDF5 file
        h5file = h5py.File(filename, 'r')
        sdf_t = np.array(h5file['sdf_vid']).transpose(3, 2, 1, 0).astype(np.float32)
        h5file.close()
    else:
        # load matlab file
        sdf_t = np.asarray([sio.loadmat(filename)['sdf_vid']], dtype=np.float32)
        sdf_t = np.squeeze(sdf_t)

    frames = []  # or "scenes"
    for t, sdf in enumerate(tqdm(sdf_t, desc="sequence {}".format(sequence_id),
                                 leave=False)):
    
        # create coordinate grid
        pixel_coords = np.stack(np.mgrid[:sdf.shape[0], :sdf.shape[1],
                                         :sdf.shape[2]],
                                axis=-1)[None, ...].astype(np.single)
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
        samples = np.zeros((sdf.shape[0]*sdf.shape[1]*sdf.shape[2], 4), 
                            dtype=np.single)
        samples[:, 0:3] = pixel_coords
        samples[:, 3] = sdf.flatten()
    
        sample_size = int(samples.shape[0] * fraction)
        
        # separate inner and outer points
        index_ip = np.where(samples[:,3] <= 0.3)
        ip_arr = samples[index_ip,:]
        index_op = np.where(samples[:,3] > 0.3)
        op_arr = samples[index_op,:]  
        
        # 70% of samples will be inner points
        no_ip = int(sample_size * 0.7)
        if len(index_ip[0]) < no_ip:
            no_ip = len(index_ip[0])
            samples_ip = ip_arr
        else:
            ip_rnd_idx = np.random.randint(0, ip_arr.shape[1], (no_ip,))
            samples_ip = np.take(ip_arr, ip_rnd_idx, 1)
            
        no_op = sample_size - no_ip
        
        op_rnd_idx = np.random.randint(0, op_arr.shape[1], (no_op,))      
        samples_op = np.take(op_arr, op_rnd_idx, 1)
    
        samples = torch.from_numpy(np.squeeze(
            np.concatenate((samples_ip, samples_op), 1)))
        
        frames.append((samples, t, filename, sequence_id))
   
    # time coordinates
    t = np.asarray([frame[1] for frame in frames], dtype=np.single)
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    
    frames = [(scene[0], t[i], *scene[2:]) for i, scene in enumerate(frames)] 

    return frames


def create_SDF(decoder, latent_vec, time, rot_ang, pixel_coords, 
               N=[128,128,128], max_batch=64**3, offset=None, 
               scale=None):

    # pre-allocate array
    num_samples = N[0]*N[1]*N[2] 
    samples = torch.zeros(num_samples, 5, dtype=torch.float32, requires_grad=False)
    
    # rotate cootdinates
    pixel_coords = torch.from_numpy(pixel_coords) @ \
        euler_angles_to_matrix(torch.from_numpy(rot_ang), 'XYZ')
    
    # paste grid into the samples array
    samples[:, 0:3] = pixel_coords
    samples[:, 3] = time.expand(num_samples)

    # infer SDF samples    
    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:4].cuda()
        samples[head : min(head + max_batch, num_samples), 4] = \
            decode_sdf(decoder, latent_vec, sample_subset).squeeze(1).detach().cpu()
        head += max_batch
    
    # reshape SDF
    sdf_values = samples[:, 4].reshape(
        N[0], N[1], N[2]).detach().cpu().numpy().astype(np.float32)
    #print('min:', np.min(sdf_values),
    #      'max:', np.max(sdf_values),'\n')

    return sdf_values
