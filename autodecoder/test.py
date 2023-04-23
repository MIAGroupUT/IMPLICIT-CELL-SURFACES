#!/usr/bin/env python3

import torch
import signal
import sys
import os

from lib.workspace import *
from lib.decoder_siren import *
from lib.utils import *
from lib.data import *


def main_function(experiment_directory, test_type=None):

    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in
                                                   specs["Description"]]))

    latent_size = specs["CodeLength"]
    num_sequences = specs["NumSequences"]

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # load decoder
    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)  
    saved_model_state = torch.load(
        os.path.join(
            experiment_directory, model_params_subdir, "latest.pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval() # set the dropout and batch norm layers to eval mode
    print(decoder)
        
    # load reconstructions & latent vectors
    reconstruction_dir = get_reconstruction_dir(experiment_directory, True)
    generation_dir = get_generation_dir(experiment_directory, True)
    
    learned_vectors = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                         + '/lat_vecs.mat')['lat_vecs']], dtype=np.float32))
    learned_rot_ang = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                         + '/rot_angs.mat')['rot_ang']], dtype=np.float32))
    
    num_frames_per_sequence_to_reconstruct = specs["ReconstructionFramesPerSequence"]
    reconstruction_dims = specs["ReconstructionDims"]
    
    # size of one reconstruction batch
    rec_size = specs["ReconstructionSubsetSize"]
                       
    # prepare normalized time coordinates   
    t = np.asarray([scene for scene in 
                    range(num_frames_per_sequence_to_reconstruct)],
                   dtype='float32')
    # normalize to [-1, 1]
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    time = torch.from_numpy(t).float().cuda() 
    
    print("Inferring {} frames per sequence with resolution {}".format(
        num_frames_per_sequence_to_reconstruct,reconstruction_dims))
    
    # Reconstruction ----------------------------------------------------------
    
    if test_type == "reconstruct":    
       
        for sequence_id in range(learned_vectors.shape[0]):
            output_sdf = np.zeros([num_frames_per_sequence_to_reconstruct,
                reconstruction_dims[0], reconstruction_dims[1],
                reconstruction_dims[2]],dtype='float32')
            latent = torch.from_numpy(
                learned_vectors[sequence_id,:]).float().cuda()
            print('\nlat. vec. preview:', 
                learned_vectors[sequence_id,:5], "...")  
            print('rot. ang.:', 
                learned_rot_ang[sequence_id], "\n")
            for i in range(num_frames_per_sequence_to_reconstruct):
                filename = get_output_filename(reconstruction_dir, "rec") + \
                    "_seq_" +  str(sequence_id).zfill(3) + "_rot_" + \
                    f'{learned_rot_ang[sequence_id,0]:.3f}' + "_" \
                    f'{learned_rot_ang[sequence_id,1]:.3f}' + "_"\
                    f'{learned_rot_ang[sequence_id,2]:.3f}'
                print("Reconstructing sequence {}, frame {}...".format(
                    sequence_id, i))
                with torch.no_grad():
                    output_sdf[i,:] = create_SDF(decoder, latent, time[i], 
                                                 learned_rot_ang[sequence_id],
                                                 N=reconstruction_dims,
                                                 max_batch=rec_size)         
            # save 3D+t SDF to matlab file
            sio.savemat(filename + '.mat', {'sdf_vid':output_sdf})
            print('Saved ' + filename + '.mat')        
        
    # generate (new shapes) ---------------------------------------------------
    
    elif test_type == "generate":   
        
        # randomly generate new latent vectors
        new_vectors = np.random.normal(0.0, 0.001, size=(
            num_sequences, latent_size)).astype(np.float32)
        
        # rotation angles
        new_rot_ang = np.array([0, 0, 0])
                
        for sequence_id in range(new_vectors.shape[0]):
            output_sdf = np.zeros([num_frames_per_sequence_to_reconstruct,
                reconstruction_dims[0], reconstruction_dims[1],
                reconstruction_dims[2]],dtype='float32')
            latent = torch.from_numpy(new_vectors[sequence_id,:]).float().cuda()
            print('\nlat. vec. preview:', 
                new_vectors[sequence_id,:5], "...")  
            print('rot. ang.:', 
                new_rot_ang, "\n")
            for i in range(num_frames_per_sequence_to_reconstruct):
                filename = get_output_filename(generation_dir, "gen") + \
                    "_seq_" +  str(sequence_id).zfill(3) + "_rot_" + \
                    f'{new_rot_ang[0]:.3f}' + "_" \
                    f'{new_rot_ang[1]:.3f}' + "_"\
                    f'{new_rot_ang[2]:.3f}'
                print("Reconstructing sequence {}, frame {}...".format(sequence_id, i))
                with torch.no_grad():
                    output_sdf[i,:] = create_SDF(decoder, latent, time[i], 
                                                 new_rot_ang,
                                                 N=reconstruction_dims,
                                                 max_batch=rec_size)         
            # save 3D+t SDF to matlab file
            sio.savemat(filename + '.mat', {'sdf_vid':output_sdf})
            print('Saved ' + filename + '.mat') 
    # -------------------------------------------------------------------------
        
    print("Done!")

if __name__ == "__main__":

    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Reconstruct")
    arg_parser.add_argument(
        "--experiment",
        "-e",
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
    
    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.test_type)
    
