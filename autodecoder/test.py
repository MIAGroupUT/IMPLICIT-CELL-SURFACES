#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
from pytorch3d.transforms import euler_angles_to_matrix # 3D rotation matrix
import signal
import sys
import os
import math
import json
import pdb
import lib
from lib.workspace import *
#from lib.models.decoder_relu import *
from lib.models.decoder_siren import *
from lib.utils import *
from lib.mesh import *



def main_function(experiment_directory, test_type=None):

    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    #data_source = specs["DataSource"]
    #train_split_file = specs["TrainSplit"]
    latent_size = specs["CodeLength"]
    
    # number of time-evolving shapes
    num_sequences = specs["NumSequences"]

    # lr_schedules = get_learning_rate_schedules(specs)

    # grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)

    # def save_latest(epoch):
    #     save_model(experiment_directory, "latest.pth", decoder, epoch)
    #     save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
    #     save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    # def adjust_learning_rate(lr_schedules, optimizer, epoch):

    #     for i, param_group in enumerate(optimizer.param_groups):
    #         param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    # scene_per_batch = specs["ScenesPerBatch"]
    # clamp_dist = specs["ClampingDistance"]
    # minT = -clamp_dist
    # maxT = clamp_dist
    # enforce_minmax = True

    # do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    # code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    # code_bound = get_spec_with_default(specs, "CodeBound", None)
    
    
    # load decoder
    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"])
    #print("training with {} GPU(s)".format(torch.cuda.device_count()))
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

    # total number of frames across all sequences
    #num_sequences = specs["NumSequences"]

    print(decoder)

    # load latent vectors
    # lat_vecs = torch.nn.Embedding(num_sequences, latent_size).cuda()
    
    # latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
    # latent_filename = os.path.join(
    #     latent_codes_dir, "latest.pth"
    # )
    # latent = torch.load(latent_filename)["latent_codes"]["weight"]
  
    # print(
    #     "Number of decoder parameters: {}".format(
    #         sum(p.data.nelement() for p in decoder.parameters())
    #     )
    # )
    # print(
    #     "Number of shape code parameters: {} (# codes {}, code dim {})".format(
    #         lat_vecs.num_embeddings * lat_vecs.embedding_dim,
    #         lat_vecs.num_embeddings,
    #         lat_vecs.embedding_dim,
    #     )
    # )
        
    # load reconstructions & latent vectors
    reconstruction_dir = get_reconstruction_dir(experiment_directory, True)
    generation_dir = get_generation_dir(experiment_directory, True)
    
    learned_vectors = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                         + '/latent_vecs.mat')['lat_vecs']], dtype=np.float32))
    learned_rot_ang = np.squeeze(np.asarray([sio.loadmat(experiment_directory 
                         + '/rot_ang.mat')['rot_ang']], dtype=np.float32))
    
    num_frames_per_sequence_to_reconstruct = specs["ReconstructionFramesPerSequence"]
    reconstruction_dims = specs["ReconstructionDims"]
    
    # size of one reconstruction batch
    rec_size = specs["ReconstructionSubsetSize"]
                       
    # prepare normalized time coordinates   
    t = np.asarray([scene for scene in range(num_frames_per_sequence_to_reconstruct)],
        dtype='float32')
    # normalize to [-1, 1]
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    time = torch.from_numpy(t).float().cuda() 
    
    # run a specified test (test_type)
    
    print("Predicting {} frames per sequence with resolution {}".format(
        num_frames_per_sequence_to_reconstruct,reconstruction_dims))
    
    # Reconstruction ----------------------------------------------------------
    
    if test_type == "reconstruct":    
       
        for sequence_id in range(learned_vectors.shape[0]):
            output_sdf = np.zeros([num_frames_per_sequence_to_reconstruct,
                reconstruction_dims[0], reconstruction_dims[1],
                reconstruction_dims[2]],dtype='float32')
            latent = torch.from_numpy(learned_vectors[sequence_id,:]).float().cuda()
            print('\nlat. vec. preview:', 
                learned_vectors[sequence_id,:5], "...")  
            print('rot. ang.:', 
                learned_rot_ang, "\n")
            for i in range(num_frames_per_sequence_to_reconstruct):
                filename = get_output_filename(reconstruction_dir, "rec") + \
                    "_seq_" +  str(sequence_id).zfill(3) + "_rot_" + \
                    f'{learned_rot_ang[0]:.3f}' + "_" \
                    f'{learned_rot_ang[1]:.3f}' + "_"\
                    f'{learned_rot_ang[2]:.3f}'
                print("Reconstructing sequence {}, frame {}...".format(sequence_id, i))
                with torch.no_grad():
                    output_sdf[i,:] = create_SDF(decoder, latent, time[i], 
                                                 learned_rot_ang,
                                                 N=reconstruction_dims)         
            # save 3D+t SDF to matlab file
            sio.savemat(filename + '.mat', {'sdf_vid':output_sdf})
            print('Saved ' + filename + '.mat')        
        
    # generate (new shapes) ---------------------------------------------------
    
    elif test_type == "generate":   
        
        # GENERATE LATENT VECTORS
        # randomly generate new latent vectors
        new_vectors = np.random.normal(0.0, 0.01, size=(num_sequences, latent_size)).astype(np.float32)
        
        # GENERATE ROTATIONS
        # generate combinations of rotation angles in radians [N, 3]
        # minimal range of Euler angles (phi,theta,psi) to cover the span
        # of rotations is the set [0,2pi)×[−pi/2,pi/2]×[0,2pi)
        phi_range = [0, 2*math.pi]
        theta_range = [-math.pi/2, math.pi/2]
        psi_range = [0, 2*math.pi]
        
        num_rot_steps = 2

        phi = np.linspace(phi_range[0], phi_range[1], num=num_rot_steps,
                          endpoint=False, dtype=np.float32)
        theta = np.linspace(theta_range[0], theta_range[1], num=num_rot_steps,
                            endpoint=True, dtype=np.float32)
        psi = np.linspace(psi_range[0], psi_range[1], num=num_rot_steps,
                          endpoint=False, dtype=np.float32)
        new_rot_ang = np.stack(np.meshgrid(phi,theta,psi), axis=-1)     
        new_rot_ang = new_rot_ang.reshape(-1, new_rot_ang.shape[-1])       
        
        for sequence_id in range(learned_vectors.shape[0]):
            output_sdf = np.zeros([num_frames_per_sequence_to_reconstruct,
                reconstruction_dims[0], reconstruction_dims[1],
                reconstruction_dims[2]],dtype='float32')
            latent = torch.from_numpy(new_vectors[sequence_id,:]).float().cuda()
            print('\nlat. vec. preview:', 
                new_vectors[sequence_id,:5], "...")  
            
            for rot_idx in range(new_rot_ang.shape[0]):
                
                print('rot. ang.:', 
                    new_rot_ang[rot_idx], "\n")
                
                for i in range(num_frames_per_sequence_to_reconstruct):
                    filename = get_output_filename(generation_dir, "gen") + \
                        "_seq_" +  str(sequence_id).zfill(3) + "_rot_" + \
                        str(rot_idx).zfill(3) + "_" + \
                        f'{new_rot_ang[rot_idx,0]:.3f}' + "_" \
                        f'{new_rot_ang[rot_idx,1]:.3f}' + "_"\
                        f'{new_rot_ang[rot_idx,2]:.3f}'
                    print("Generating sequence {}, frame {}...".format(sequence_id, i))
                    with torch.no_grad():
                        output_sdf[i,:] = create_SDF(decoder, latent, time[i], 
                                                     new_rot_ang[rot_idx],
                                                     N=reconstruction_dims)         
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
    
    # add cmd parameter for test_type
    
    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.test_type)
    #main_function('experiments/celegans', "reconstruct")