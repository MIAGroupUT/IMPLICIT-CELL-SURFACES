'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
import torch
#from functools import partial
import skvideo.datasets

import scipy.io as sio # matlab saving
import numpy as np # matlab saving

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
#p.add_argument('--batch_size', type=int, default=1)
#p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
#p.add_argument('--num_epochs', type=int, default=100000,
#               help='Number of epochs to train for.')

#p.add_argument('--epochs_til_ckpt', type=int, default=1000,
#               help='Time interval in seconds until checkpoint is saved.')
#p.add_argument('--steps_til_summary', type=int, default=100,
#               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset', type=str, default='bikes',
               help='Video dataset; one of (cat, bikes)', choices=['cat', 'bikes'])
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=0.01, # 1 / (50*4)
               help='What fraction of video pixels to sample in each batch (default is all)')

p.add_argument('--checkpoint_path', required=True, help='Checkpoint to trained model.')
opt = p.parse_args()

if opt.dataset == 'cat':
    #video_path = './data/video_512.npy'
    #video_path = './data/cat_video.mp4'
    video_path = './data/cell_00702_vid_z32.mat'
elif opt.dataset == 'bikes':
    video_path = skvideo.datasets.bikes()

vid_dataset = dataio.Video(video_path)
coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=opt.sample_frac, training=False)
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)

# Define the model and load in checkpoint path
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=4, out_features=vid_dataset.channels,
                                 mode='mlp', hidden_features=1024, num_hidden_layers=3)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=4, out_features=vid_dataset.channels, mode=opt.model_type)
else:
    raise NotImplementedError
model.load_state_dict(torch.load(opt.checkpoint_path)) # load checkpoint
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# Define the loss
#loss_fn = partial(loss_functions.image_mse, None)
#summary_fn = partial(utils.write_video_summary, vid_dataset)

#training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
#               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
#               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)

data_dims = vid_dataset.shape
data_len = int(np.prod(data_dims))
data_frac = opt.sample_frac
sample_size = int(data_len * data_frac)
iters = int(data_len / sample_size)

sdf_fit = np.zeros(data_len, dtype='float32')

head = 0
for i in range(iters):
    # get input data
    model_input, _ = next(iter(dataloader))
    model_input = {key: value.cuda() for key, value in model_input.items()}

    # Evaluate the trained model
    with torch.no_grad():
        model_output = model(model_input)

    sdf_fit[head : min(head + sample_size, data_len)] = torch.squeeze(model_output['model_out']).detach().cpu().numpy().astype(np.float32)
    
    print('Infering batch', i+1, 'of', iters)
    
    head += sample_size


sdf_fit = sdf_fit.reshape((data_dims[0],data_dims[1],data_dims[2],data_dims[3]))

directory = './generated_data'
# create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
        
filename = 'sdf_vid_fit' + '.mat' 

# save .mat file
sio.savemat(directory + '/' + filename, {'sdf_vid': sdf_fit})

print('File', filename, 'saved.')

