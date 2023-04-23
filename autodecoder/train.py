#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
from pytorch3d.transforms import euler_angles_to_matrix # 3D rotation matrix
import signal
import sys
import math

from lib.workspace import *
from lib.decoder_siren import *
from lib.utils import *
from lib.data import *


def main_function(experiment_directory):

    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in 
                                                   specs["Description"]]))

    data_source = specs["DataSource"]
    latent_size = specs["CodeLength"]
    
    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)

    def save_latest(epoch):
        print("Saving checkpoint...")
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
        
    signal.signal(signal.SIGINT, signal_handler)

    sample_fraction = specs["SampleFraction"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
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

    sdf_dataset = InMemorySDFSamplesFraction(data_source, sample_fraction)

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    print("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
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
    
    rot_ang = torch.nn.Embedding(num_sequences, 3).cuda()
    torch.nn.init.normal_(
        rot_ang.weight.data,
        0.0,
        (math.pi**2)/64,
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
            },
        ]
    )

    loss_log = []

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
    
    # training loop
    for epoch in range(start_epoch, num_epochs + 1):

        decoder.train()
        if epoch > lr_schedules[2].interval:
            adjust_learning_rate(lr_schedules, optimizer_dec_lat, epoch)
        else:
            adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        
        running_loss=0

        for sdf_data, times, _, sequence_ids in sdf_loader:

            optimizer_all.zero_grad()
            optimizer_dec_lat.zero_grad()
                
            num_samp_per_frame = sdf_data.shape[1]
            num_sdf_samples = sdf_data.shape[0] * sdf_data.shape[1]
            sdf_data.requires_grad = False
            
            # rotate coordinates 
            xyz_rot = sdf_data[:, :, 0:3].cuda() @ \
                euler_angles_to_matrix(rot_ang(sequence_ids.cuda()), 'XYZ')           
            
            xyz_rot = xyz_rot.reshape(-1, 3)    
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

            if do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = 1000 * (
                    code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                ) / num_sdf_samples

                batch_loss = batch_loss + reg_loss.cuda()

            batch_loss.backward()

            loss_log.append(batch_loss.item())
            
            running_loss += batch_loss.item()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
 
            if epoch > lr_schedules[2].interval:
                optimizer_dec_lat.step()
            else:
                optimizer_all.step()

        train_loss=running_loss/len(sdf_loader)
        print("epoch {} / {}, loss {:.8f}, ".format(epoch, num_epochs,
                                                    train_loss), end='')
        
        # save progress
        if epoch % log_frequency == 0 or epoch == num_epochs:
            save_latest(epoch)

            # save latent vectors to .mat file
            learned_vectors = np.zeros([lat_vecs.num_embeddings, 
                                        lat_vecs.embedding_dim],dtype='float32')
            for i in range(lat_vecs.num_embeddings):
                learned_vectors[i,:] = \
                lat_vecs(torch.tensor(i).cuda()).squeeze(0).detach().cpu().numpy().astype(np.float32)
            sio.savemat(experiment_directory + '/lat_vecs.mat', {'lat_vecs':learned_vectors})
            
            # save rotation angles to .mat file
            learned_rot_ang = np.zeros([rot_ang.num_embeddings, 
                                        rot_ang.embedding_dim],dtype='float32')
            for i in range(rot_ang.num_embeddings):
                learned_rot_ang[i,:] = \
                rot_ang(torch.tensor(i).cuda()).squeeze(0).detach().cpu().numpy().astype(np.float32)
            sio.savemat(experiment_directory + '/rot_angs.mat', {'rot_ang':learned_rot_ang}) 
            
    print("Done!")

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )

    args = arg_parser.parse_args()
    main_function(args.experiment_directory)
    
