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
    latent_size = specs["CodeLength"]
    latent_size2 = specs["CodeLengthRotation"]
    
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
    decoder = DeepSDF(latent_size, latent_size2, **specs["NetworkSpecs"]).cuda()

    print("training with {} GPU(s)".format(torch.cuda.device_count()))
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 1)

    sdf_dataset = lib.data.InMemorySDFSamplesFraction(sample_fraction,
                                                      no_rotations=specs["NumRotations"])

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
    num_shapes_total = specs["NumRotations"] # TODO only a number of rotations for each class
    ###num_sequences = specs["NumSequences"]
    num_shape_classes = specs["NumShapeClasses"]

    print("There are total of {} shapes".format(num_shapes_total))
    print(decoder)

    lat_vecs = torch.nn.Embedding(num_shape_classes, latent_size).cuda()
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    lat_vecs2 = torch.nn.Embedding(num_shapes_total, latent_size2).cuda()
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size2),
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    
    optimizer_dec_latvecs = torch.optim.Adam(
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
                "params": lat_vecs2.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
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
    print(
        "Number of shape rotation parameters: {} (# codes {}, code dim {})".format(
            lat_vecs2.num_embeddings * lat_vecs2.embedding_dim,
            lat_vecs2.num_embeddings,
            lat_vecs2.embedding_dim,
        )
    )
    
    # train parameterization
    for epoch in range(start_epoch, num_epochs + 1):

        decoder.train()
        adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        
        running_loss=0
        
        #print(sdf_dataset[0])
        
        #exit()

        #for sdf_data, indices, filenames in sdf_loader:
        # TODO add shape_class and use it for fixing batch_vecs = (line 203)
        for sdf_data, shape_class, shape_id in sdf_loader:
            
            optimizer_dec_latvecs.zero_grad()
            optimizer_all.zero_grad()
            #print(sdf_data.shape)
            num_samp_per_scene = sdf_data.shape[1]

            sdf_data = sdf_data.reshape(-1, 3)
            # number of samples * batch size
            num_sdf_samples = sdf_data.shape[0] # num_samp_per_scene * batch_size
            sdf_data.requires_grad = False

            #xyzt = torch.cat((sdf_data[:, 0:3], times.unsqueeze(-1).repeat(1, num_samp_per_frame).view(-1, 1)), dim=1)
            #sdf_gt = sdf_data[:, 3].unsqueeze(1)
            xy = sdf_data[:, 0:2]
            sdf_gt = sdf_data[:, 2].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            batch_vecs = lat_vecs(shape_class.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).cuda())
            batch_vecs2 = lat_vecs2(shape_id.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1).cuda())
           
            pred_sdf = decoder(batch_vecs, batch_vecs2, xy.cuda())
            
            if enforce_minmax:
                pred_sdf = torch.clamp(pred_sdf, minT, maxT)

            batch_loss = loss_l1(pred_sdf, sdf_gt.cuda()) / num_sdf_samples

            if do_code_regularization:
                l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                reg_loss = (
                    code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                ) / num_sdf_samples
                
                l2_size_loss2 = torch.sum(torch.norm(batch_vecs2, dim=1))
                reg_loss = (
                    code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                ) / num_sdf_samples
                reg_loss2 = (
                    code_reg_lambda * min(1, epoch / 100) * l2_size_loss2
                ) / num_sdf_samples

                #batch_loss = batch_loss + reg_loss.cuda() #+ reg_loss2.cuda()
                batch_loss = batch_loss + reg_loss.cuda() + reg_loss2.cuda()

            batch_loss.backward()

            # item() method extracts the loss’s value as a Python float
            loss_log.append(batch_loss.item())
            
            running_loss += batch_loss.item()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            #optimizer_dec_latvecs.step()
            optimizer_all.step()

        train_loss=running_loss/len(sdf_loader)
        print("epoch {} / {}, loss {:.8f}".format(epoch, num_epochs, train_loss))        

        # save progress
        if epoch % log_frequency == 0 or epoch == num_epochs:
            save_latest(epoch)

            # save latent vectors to .mat file
            learned_vectors = np.zeros([lat_vecs.num_embeddings, lat_vecs.embedding_dim],dtype='float32')

            for _, shape_class, _ in sdf_loader_reconstruction:
                latent = lat_vecs(shape_class.cuda()).squeeze(0)
                learned_vectors[shape_class,:] = latent.detach().cpu().numpy().astype(np.float32)

            sio.savemat(experiment_directory + '/latent_vecs.mat', {'lat_vecs':learned_vectors})    
            
            learned_vectors2 = np.zeros([lat_vecs2.num_embeddings, lat_vecs2.embedding_dim],dtype='float32')

            for _, _, shape_id in sdf_loader_reconstruction:
                latent2 = lat_vecs2(shape_id.cuda()).squeeze(0)
                learned_vectors2[shape_id,:] = latent2.detach().cpu().numpy().astype(np.float32)

            sio.savemat(experiment_directory + '/latent_vecs_rot.mat', {'lat_vecs_rot':learned_vectors2}) 
            
    print("Done!")

if __name__ == "__main__":

    # import argparse

    # arg_parser = argparse.ArgumentParser(description="Train")
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
