#!/usr/bin/env python3

import torch
import signal
import sys
import os
from math import pi
import time

from lib.workspace import *
from lib.decoder import *
from lib.utils import *
from lib.data import *


def main_function(experiment_directory, test_type=None, test_epoch=None):

    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in
                                                   specs["Description"]]))

    latent_size = specs["CodeLength"]
    num_sequences = specs["NumSequences"]
    # size of one reconstruction batch
    rec_size = specs["ReconstructionSubsetSize"]
    num_frames_per_sequence_to_reconstruct = specs["ReconstructionFramesPerSequence"]
    reconstruction_dims = specs["ReconstructionDims"]
    # use .MAT (False) or .HDF5 (True) to load and save SDFs
    use_hdf5 = specs["UseHDF5"] 
    # gzip compression level: 0 (min) ... 9 (max)
    hdf5_compr = specs["HDF5CompressionLevel"] 

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # load decoder
    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)  
    saved_model_state = torch.load(
        os.path.join(
            experiment_directory, model_params_subdir, 'e' + \
                test_epoch.zfill(4) + '.pth'
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval() # set the dropout and batch norm layers to eval mode
    print(decoder)
                                     
    # prepare normalized time coordinates in [-1, 1]
    t = np.asarray([scene for scene in 
                    range(num_frames_per_sequence_to_reconstruct)],
                   dtype='float32')
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    t = torch.from_numpy(t)
    
    
    def get_pixel_coords(N):
        # prepare coordinate grid
        pixel_coords = np.stack(np.mgrid[:N[0], :N[1], :N[2]], 
                                axis=-1)[None, ...].astype(np.single)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(N[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (N[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (N[2] - 1)
        pixel_coords = np.squeeze(pixel_coords)
        
        # normalize from [0, 1] to [-1, 1]
        pixel_coords -= 0.5
        pixel_coords *= 2.  
        
        # flatten all but the last dimension
        pixel_coords = pixel_coords.reshape(-1, pixel_coords.shape[-1])
        
        return pixel_coords
    
    
    def save_sdf(output_sdf, filename):
        if use_hdf5:
            print('Saving and compressing HDF5 file with gzip level {}...'.format(
                hdf5_compr)) 
            # save 3D+t SDF to HDF5 file
            output_sdf = np.transpose(output_sdf,(3, 2, 1, 0))
            h5file = h5py.File(filename + '.h5', 'w')
            dset = h5file.create_dataset("sdf_vid", shape=output_sdf.shape,
                                         data=output_sdf,
                                         fillvalue=0.0,
                                         chunks=(output_sdf.shape[0],
                                                 output_sdf.shape[1],
                                                 output_sdf.shape[2],
                                                 1),
                                         dtype=np.float32,
                                         compression='gzip',
                                         compression_opts=hdf5_compr)
            h5file.close()
            print('Saved ' + filename + '.h5')  
        
        else:
            # save 3D+t SDF to matlab file
            sio.savemat(filename + '.mat', {'sdf_vid':output_sdf})
            print('Saved ' + filename + '.mat')
   
    
    def infer_sdf(latent_vectors, rot_angs, t, fprefix):
        pixel_coords = get_pixel_coords(reconstruction_dims)
        output_sdf = np.zeros([t.shape[0],
            reconstruction_dims[0], reconstruction_dims[1],
            reconstruction_dims[2]],dtype='float32')
        for sequence_id in range(latent_vectors.shape[0]):
            infer_start_t = time.time()
            latent = torch.from_numpy(
                latent_vectors[sequence_id,:]).float().cuda()
            print("\n--> Time-evolving shape {} of {}".format(
                sequence_id+1, latent_vectors.shape[0]))
            print("---------------------------------")
            print('rotation angles:\n', 
                rot_angs[sequence_id])
            print('latent vector preview:\n', 
                latent_vectors[sequence_id,:8], "...")
            filename = fprefix + \
                "_seq_" +  str(sequence_id).zfill(3) + "_rot_" + \
                f'{rot_angs[sequence_id,0]:.3f}' + "_" \
                f'{rot_angs[sequence_id,1]:.3f}' + "_"\
                f'{rot_angs[sequence_id,2]:.3f}'
            for i in range(t.shape[0]):               
                print("Inferring time point {}...".format(i))
                with torch.no_grad():
                    output_sdf[i,:] = create_SDF(decoder, latent, t[i], 
                                                 rot_angs[sequence_id],
                                                 pixel_coords,
                                                 N=reconstruction_dims,
                                                 max_batch=rec_size) 
            infer_t = time.time() - infer_start_t
            print("Inference time: {:.1f}s ".format(infer_t))
            save_start_t = time.time()
            save_sdf(output_sdf, filename)
            save_t = time.time() - save_start_t
            print("Save time: {:.1f}s ".format(save_t))
    
    print("\nInferring {} time-evolving shapes".format(num_sequences) + \
          " with {} time points".format(t.shape[0]) + \
          " and grid resolution of {}!".format(reconstruction_dims))
    
    
    # Reconstruct #############################################################
    
    if test_type == "reconstruct":  
    
        # load reconstructions & latent vectors
        if test_epoch is None:
            learned_vectors = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                + '/' + latent_codes_subdir + '/' + \
                    'lat_vecs.mat')['lat_vecs']], dtype=np.float32))
            learned_rot_angs = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                + '/' + latent_codes_subdir + '/' + \
                    'rot_angs.mat')['rot_ang']], dtype=np.float32))  
        else:
            learned_vectors = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                + '/' + latent_codes_subdir + '/e' + test_epoch.zfill(4) + \
                    '_lat_vecs.mat')['lat_vecs']], dtype=np.float32))
            learned_rot_angs = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                + '/' + latent_codes_subdir + '/e' + test_epoch.zfill(4) + \
                    '_rot_angs.mat')['rot_ang']], dtype=np.float32))
                
        fprefix = get_output_filename(get_reconstruction_dir(
            experiment_directory, True), "rec")
       
        infer_sdf(learned_vectors, learned_rot_angs, t, fprefix)       
        
        
    # Generate (new shapes) ###################################################
    
    elif test_type == "generate":   
        
        # randomly generate new latent vectors
        new_vectors = np.random.normal(0.0, 0.001, size=(
            num_sequences, latent_size)).astype(np.float32)
        
        # randomly generate new rotation angles
        new_rot_ang = np.random.uniform(low=-pi, high=pi, 
            size=(num_sequences, 3)).astype(np.float32)
        
        fprefix = get_output_filename(get_generation_dir(
            experiment_directory, True), "gen")
        
        infer_sdf(new_vectors, new_rot_ang, t, fprefix)
                
            
    ###########################################################################
        
    print("\nDone!")

if __name__ == "__main__":

    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Test")
    '''arg_parser.add_argument(
        "--experiment",
        "-x",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--test_type",
        "-t",
        dest="test_type",
        required=True,
        help="Type of the test. Valid values: [reconstruct, generate]",
    )
    arg_parser.add_argument(
        "--epoch",
        "-e",
        dest="test_epoch",
        required=False,
        help="Index of the epoch.",
    )
    '''
    
    args = arg_parser.parse_args()
    #main_function(args.experiment_directory, args.test_type, args.test_epoch)
    main_function('./experiments/test', 'reconstruct', '520')
    
