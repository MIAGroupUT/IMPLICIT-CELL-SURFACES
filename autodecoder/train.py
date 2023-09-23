#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
from pytorch3d.transforms import euler_angles_to_matrix # 3D rotation matrix
import signal
import sys
import math
import time

from lib.workspace import *
from lib.decoder import *
from lib.utils import *
from lib.data import *


def main_function(experiment_directory):

    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in 
                                                   specs["Description"]]))

    data_source = specs["DataSource"]
    latent_size = specs["CodeLength"]
    opt_rot = specs["OptimizeRot"]
    
    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)
    
    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            if opt_rot and i == 1:
                param_group["lr"] = 10*lr_schedules[i].get_learning_rate(epoch)
            else:
                param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
                    
    signal.signal(signal.SIGINT, signal_handler)

    use_hdf5 = specs["UseHDF5"] # use .MAT or .HDF5 to load and save SDFs
    sample_fraction = specs["TrainSampleFraction"]
    frames_per_batch = specs["FramesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    optr_ep = specs["OptRotEpoch"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = specs["ClampSDFminmax"]

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)
    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"]).cuda()

    print("training with {} GPU(s)".format(torch.cuda.device_count()))
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 1)

    otf_sampling = specs["OnTheFlySampling"]
    sdf_dataset = SDFSamples(data_source, sample_fraction, otf_sampling, use_hdf5)

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    print("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=frames_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    sdf_loader_reconstruction = data_utils.DataLoader(
        sdf_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    # total number of frames across all sequences
    num_frames_total = len(sdf_dataset)
    num_sequences = specs["NumSequences"]

    print("There are total of {} frames across {} sequences".format(
        num_frames_total, num_sequences))
    print(decoder)

    lat_vecs = torch.nn.Embedding(num_sequences, latent_size).cuda()
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    
    if opt_rot:
        rot_ang = torch.nn.Embedding(num_sequences, 3).cuda()
        torch.nn.init.normal_(
            rot_ang.weight.data,
            0.0,
            (math.pi**2)/64,
        )
    else:
        rot_ang = torch.nn.Embedding(num_sequences, 3).cuda()
        torch.nn.init.constant_(
            rot_ang.weight.data,
            0.0
        )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": rot_ang.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            }
        ]
    )

    optimizer_dec_lat = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            }
        ]
    )

    start_epoch = 1

    print("starting from epoch {}".format(start_epoch))
    print(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    print(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    
    # prepare log file  
    logfpth = experiment_directory + '/' \
        + time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss") \
        + '_train_log.txt'
    logf = open(logfpth, "w")
    with open(experiment_directory + '/specs.json') as specf:
        logf.write(specf.read())
        logf.write('\n\n\n')
    logf.close()
    
    # training loop
    for epoch in range(start_epoch, num_epochs + 1):
        
        epoch_start_t = time.time()

        decoder.train()
        if not opt_rot or epoch > optr_ep:
            adjust_learning_rate(lr_schedules, optimizer_dec_lat, epoch)
        else:
            adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        
        running_loss_sdf = 0
        running_loss_code = 0

        for sdf_data, times, _, sequence_ids in sdf_loader:

            optimizer_dec_lat.zero_grad()
            optimizer_all.zero_grad()
                
            num_samp_per_frame = sdf_data.shape[1]
            num_sdf_samples = sdf_data.shape[0] * sdf_data.shape[1]
            sdf_data.requires_grad = False
            
            if opt_rot:
                # rotate coordinates 
                xyz_rot = sdf_data[:, :, 0:3].cuda() @ \
                    euler_angles_to_matrix(rot_ang(sequence_ids.cuda()), 'XYZ')           
                
                xyz_rot = xyz_rot.reshape(-1, 3) 
            else:
                xyz_rot = sdf_data[:, :, 0:3].cuda().reshape(-1, 3) 
                
            sdf_gt = sdf_data[:, :, 3].reshape(-1, 1)  

            xyzt = torch.cat((xyz_rot, times.cuda().unsqueeze(-1).repeat(
                1, num_samp_per_frame).view(-1, 1)), dim=1)

            batch_vecs = lat_vecs(sequence_ids.unsqueeze(-1).repeat(
                1, num_samp_per_frame).view(-1).cuda())
            
            pred_sdf = decoder(batch_vecs, xyzt)
            
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)
                pred_sdf = torch.clamp(pred_sdf, minT, maxT)

            batch_loss = loss_l1(pred_sdf, sdf_gt.cuda()) / num_sdf_samples
            
            running_loss_sdf += batch_loss.item()

            if do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                if not opt_rot:
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 1000) * l2_size_loss
                    ) / num_sdf_samples
                else:
                    reg_loss = 1000 * (code_reg_lambda * l2_size_loss \
                        * 1/(min(1, epoch / 100))) / num_sdf_samples

                batch_loss = batch_loss + reg_loss.cuda()
                
                running_loss_code += reg_loss.item()

            batch_loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
 
            if not opt_rot or epoch > optr_ep:
                optimizer_dec_lat.step()
            else:
                optimizer_all.step()

        train_loss_sdf=running_loss_sdf/len(sdf_loader)
        train_loss_code=running_loss_code/len(sdf_loader)
        
        epoch_t = time.time() - epoch_start_t
        
        # print progress
        logstr1 = "epoch {} / {}, SDF loss {:.5f}, code loss {:.5f}, time {:.1f}s ".format(
            epoch, num_epochs, train_loss_sdf, train_loss_code, epoch_t)
        print(logstr1)
        #print(logstr1, end='')
        #logstr2 = "ang: " + np.array2string(
        #    rot_ang(torch.tensor(0).cuda()).detach().cpu().numpy(), precision=5)
        #print(logstr2)       
        # log to file   
        logf = open(logfpth, "a")
        logf.write(logstr1+'\n')
        #logf.write(logstr1)
        #logf.write(logstr2+'\n')
        logf.close()
        
        # save progress
        if epoch % log_frequency == 0 or epoch == num_epochs:
            print("Saving checkpoint...")
            
            # decoder
            save_model(experiment_directory, 'e' + str(epoch).zfill(4) + \
                       '_model.pth', decoder, epoch)            
            
            # latent vectors 
            latent_codes_dir = get_latent_codes_dir(experiment_directory, True)
            learned_vectors = np.zeros([lat_vecs.num_embeddings, 
                                        lat_vecs.embedding_dim],dtype='float32')
            for i in range(lat_vecs.num_embeddings):
                learned_vectors[i,:] = \
                lat_vecs(torch.tensor(i).cuda()).squeeze(0).detach().cpu().numpy().astype(np.float32)
            sio.savemat(latent_codes_dir + '/e' + str(epoch).zfill(4) \
                        + '_lat_vecs.mat', {'lat_vecs':learned_vectors})
            
            # rotation angles
            learned_rot_ang = np.zeros([rot_ang.num_embeddings, 
                                        rot_ang.embedding_dim],dtype='float32')
            for i in range(rot_ang.num_embeddings):
                learned_rot_ang[i,:] = \
                rot_ang(torch.tensor(i).cuda()).squeeze(0).detach().cpu().numpy().astype(np.float32)
            sio.savemat(latent_codes_dir + '/e' + str(epoch).zfill(4) \
                        + '_rot_angs.mat', {'rot_ang':learned_rot_ang}) 
    
    print("Done!")

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train")
    arg_parser.add_argument(
        "--experiment",
        "-x",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )

    args = arg_parser.parse_args()
    main_function(args.experiment_directory)
    
