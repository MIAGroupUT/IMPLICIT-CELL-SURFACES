#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import math
import json
import pdb
import lib
from lib.workspace import *
from lib.models.decoder_relu import *
#from lib.models.decoder_siren import *
from lib.utils import *
from lib.mesh import *


def main_function(experiment_directory):

    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    #data_source = specs["DataSource"]
    #train_split_file = specs["TrainSplit"]
    latent_size = specs["CodeLength"]
    latent_size2 = specs["CodeLengthRotation"]

    #lr_schedules = get_learning_rate_schedules(specs)

    #grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)

    #def save_latest(epoch):
    #    save_model(experiment_directory, "latest.pth", decoder, epoch)
    #    save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
    #    save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    #def adjust_learning_rate(lr_schedules, optimizer, epoch):

    #    for i, param_group in enumerate(optimizer.param_groups):
    #        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    #scene_per_batch = specs["ScenesPerBatch"]
    #clamp_dist = specs["ClampingDistance"]
    #minT = -clamp_dist
    #maxT = clamp_dist
    #enforce_minmax = True

    #do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    #code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    #code_bound = get_spec_with_default(specs, "CodeBound", None)
    
    
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
    
    learned_vectors = np.squeeze(np.asarray([sio.loadmat(experiment_directory +
                         '/latent_vecs.mat')['lat_vecs']], dtype=np.float32))
    #learned_vectors = np.asarray([sio.loadmat(experiment_directory +
    #                     '/latent_vecs.mat')['lat_vecs']], dtype=np.float32)
    learned_vectors2 = np.squeeze(np.asarray([sio.loadmat(experiment_directory + 
                         '/latent_vecs_rot.mat')['lat_vecs_rot']], dtype=np.float32))
    #learned_vectors2 = np.asarray([sio.loadmat(experiment_directory + 
    #                     '/latent_vecs_rot.mat')['lat_vecs_rot']], dtype=np.float32)

    learned_vectors2 = np.array([learned_vectors2])
    # total number of frames across all sequences
    num_shape_classes = learned_vectors.shape[0]
    num_rotations = learned_vectors2.shape[0]
    #num_sequences = specs["NumSequences"]

    print(decoder)

    # load latent vectors
    #lat_vecs = torch.nn.Embedding(num_shape_classes, latent_size).cuda()
    #lat_vecs2 = torch.nn.Embedding(num_rotations, latent_size2).cuda()
    
    #latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
    #latent_filename = os.path.join(
    #    latent_codes_dir, "latest.pth"
    #)
    #latent = torch.load(latent_filename)["latent_codes"]["weight"]
  
    print(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    #print(
    #    "Number of shape code parameters: {} (# codes {}, code dim {})".format(
    #        lat_vecs.num_embeddings * lat_vecs.embedding_dim,
    #        lat_vecs.num_embeddings,
    #        lat_vecs.embedding_dim,
    #    )
    #)
        
    # store reconstructions & latent vectors
    decoder.eval() # set the dropout and batch norm layers to eval mode
    reconstruction_dir = get_reconstruction_dir(experiment_directory, True)
    interpolation_dir = get_interpolation_dir(experiment_directory, True)
    
    #learned_vectors = np.squeeze(np.asarray([sio.loadmat(experiment_directory + '/latent_vecs.mat')['lat_vecs']], dtype=np.float32))
    
    #num_frames_per_sequence_to_reconstruct = specs["ReconstructionFramesPerSequence"]
    reconstruction_dims = specs["ReconstructionDims"]
    
    # Reconstruction -----------------------------------------------------------
    # print("Reconstructing {} shapes with resolution {}".format(num_shape_classes,reconstruction_dims))
    
    # # size of one reconstruction batch
    # rec_size = specs["ReconstructionSubsetSize"]
                       
    # # prepare normalized time coordinates   
    # #t = np.asarray([scene for scene in range(num_frames_per_sequence_to_reconstruct)],
    # #    dtype='float32')
    # # normalize to [-1, 1]
    # #t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    # #time = torch.from_numpy(t).float().cuda() 
   
    # for shape_class in range(learned_vectors.shape[0]):
    #     #for rotation in range(learned_vectors2.shape[0]):
    #     output_sdf = np.zeros([reconstruction_dims[0], reconstruction_dims[1]],
    #         dtype='float32')
    #     latent = torch.from_numpy(learned_vectors[shape_class,:]).float().cuda()
    #     #latent2 = torch.from_numpy(learned_vectors2[rotation,:]).float().cuda()
    #     #latent = torch.from_numpy(np.expand_dims(learned_vectors,0)[shape_class,:]).float().cuda()
    #     latent2 = torch.from_numpy(np.expand_dims(learned_vectors2[shape_class],0)).float().cuda()
    #     print('\nlat. vec. preview:', 
    #         latent[:5].detach().cpu().numpy().astype(np.float32), "...\n")   
    #     print('\nrot. vec. preview:', 
    #         latent2[:5].detach().cpu().numpy().astype(np.float32), "...\n") 
       
    #     filename = get_output_filename(reconstruction_dir, "rec") + \
    #         "_id_" + str(shape_class).zfill(3) + "_rot_" + \
    #         str(latent2[0].detach().cpu().numpy().astype(np.float32)).zfill(3)
    #     print("Reconstructing shape {}...".format(shape_class))
    #     with torch.no_grad():
    #         output_sdf = create_SDF(decoder, latent, latent2,
    #                                      N=reconstruction_dims,
    #                                      max_batch=rec_size)         
    #     # save SDF to matlab file
    #     sio.savemat(filename + '.mat', {'sdf_norm':output_sdf})
    #     print('Saved ' + filename + '.mat')        

    # print("Done!")
    
    
    # Generating new rotations ------------------------------------------------
    #from math import pi
    
    rotations_rad = np.linspace(-2*math.pi, 2*math.pi, num=50, endpoint=True, dtype=np.float32)
    #rotations_rad = np.array([math.pi/2]) # reproduces the issue
    #rotations_rad = np.zeros(1) # works
    
    print("\n\n\n\nReconstructing {} shapes with resolution {}".format(num_shape_classes,reconstruction_dims))
    
    # size of one reconstruction batch
    rec_size = specs["ReconstructionSubsetSize"]
                       
    # prepare normalized time coordinates   
    #t = np.asarray([scene for scene in range(num_frames_per_sequence_to_reconstruct)],
    #    dtype='float32')
    # normalize to [-1, 1]
    #t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    #time = torch.from_numpy(t).float().cuda() 
   
    
    for shape_class in range(learned_vectors.shape[0]):
        for rot_idx in range(rotations_rad.size):
            output_sdf = np.zeros([reconstruction_dims[0], reconstruction_dims[1]],
                dtype='float32')
            latent = torch.from_numpy(learned_vectors[shape_class,:]).float().cuda()
            #latent2 = torch.from_numpy(learned_vectors2[rotation,:]).float().cuda()
            #latent = torch.from_numpy(np.expand_dims(learned_vectors,0)[shape_class,:]).float().cuda()
            #latent2 = torch.from_numpy(np.expand_dims(learned_vectors2[shape_class],0)).float().cuda()
            '''latent2
            tensor([-1.0541], device='cuda:0')

            latent2.shape
            torch.Size([1])'''
            print('\nlat. vec. preview:', 
                latent[:5].detach().cpu().numpy().astype(np.float32), "...\n")   
            print('\nrot. vec. preview:', 
                rotations_rad[rot_idx], "...\n") 
           
            filename = get_output_filename(reconstruction_dir, "new") + \
                "_id_" + str(shape_class).zfill(3) + "_" + str(rot_idx).zfill(3) + "_rot_" + \
                str(np.around(math.degrees(rotations_rad[rot_idx])).astype(np.int32)).zfill(3)
            print("Reconstructing shape {}...".format(shape_class))
            with torch.no_grad():
                output_sdf = create_SDF(decoder, latent,
                                        rotations_rad[rot_idx],
                                        N=reconstruction_dims,
                                        max_batch=rec_size)         
            # save SDF to matlab file
            sio.savemat(filename + '.mat', {'sdf_norm':output_sdf})
            print('Saved ' + filename + '.mat') 
            rot_idx += 1

if __name__ == "__main__":

    # import argparse

    # arg_parser = argparse.ArgumentParser(description="Reconstruct")
    # arg_parser.add_argument(
    #     "--experiment",
    #     "-e",
    #     dest="experiment_directory",
    #     required=True,
    #     help="The experiment directory. This directory should include "
    #     + "experiment specifications in 'specs.json', and logging will be "
    #     + "done in this directory as well.",
    # )

    # args = arg_parser.parse_args()
    # main_function(args.experiment_directory)
    main_function('experiments/shapegen')
