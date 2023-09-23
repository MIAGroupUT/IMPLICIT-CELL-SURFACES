close all
clear
%clc

input_dir = './input_data';
output_dir = './train_sdf';

clamp_dist = 32; % SDF clamping distance
normalize = true; % normalize SDF values to [-1,1]

use_HDF5 = true; % set output to MAT or HDF5 files
clevel = 9; % gzip compression level for HDF5 (0, ..., 9)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% add path to functions for current MATLAB session
addpath('./src');

% create output dir if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% for each subdirectory
input_dirs = dir(input_dir);
for sd = 3:length(input_dirs) 
    
    % get subdir name
    subdir_name = input_dirs(sd).name;
    
    % list of input mask files
    input_mask_list = dir(strcat(input_dir, '/', subdir_name, ...
        '/phantom/sim_data_afterTM/mask_t*.tif'));
    
    % process each sample in the time-lapse sequence
    sdf_vid = zeros([length(input_mask_list) 256 256 256], 'single');
    for i = 1:length(input_mask_list)

        input_img = strcat(input_dir, '/', subdir_name, ...
            '/phantom/sim_data_afterTM/', input_mask_list(i).name);

        % load 3D volumes from TIF images
        info = imfinfo(input_img);
        num_slices = numel(info);
        mask_lab = zeros(info(1).Height, info(1).Width, num_slices, ...
                         'uint16');
        for s = 1:num_slices
            mask_lab(:,:,s) = imread(input_img, s);
        end        

        % resize and pad
        mask_res = imresize3(mask_lab, 0.65, 'nearest');
        mask_pad = padarray(mask_res, [150 150 150], 0, 'both'); 

        % crop and center according to main cell body
        desired_size = [256 256 256];
        [rows, columns, slices] = ...
            determine_crop_section_3d(mask_pad == 1, desired_size);
        mask_bin = mask_pad(rows, columns, slices) > 0;

        % keep only the biggest connected component
        mask_bin = get_biggest_component(mask_bin);
        
        % Compute SDF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % compute distances and clamp
        edf_out = bwdist(mask_bin); % outer dist (pos)
        clamp_idx = edf_out > clamp_dist;
        edf_out(clamp_idx) = clamp_dist; 
        edf_in = bwdist(~mask_bin); % inner dist (neg)
        
        % combine
        sdf = edf_out - edf_in;
        
        % normalize
        if normalize; sdf = sdf ./ max(abs(sdf(:))); end
        
        % Save time point %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        sdf_vid(i, :, :, :) = sdf;
        
        % save preview
        [~, filename, ext] = fileparts(input_img);
        output_prev_dir = strcat(output_dir, '/preview/', subdir_name);
        % create preview dir if it doesn't exist
        if ~exist(output_prev_dir, 'dir')
            mkdir(output_prev_dir);
        end       
        imwrite(uint8(max(sdf<=0,[],3))*255, ...
            strcat(output_prev_dir, '/', filename, '.png'));
        
        fprintf('Processed %s%s\n', filename, ext);

    end
    
    % save 3D+time SDF
    fpath = strcat(output_dir, '/', subdir_name);
    if use_HDF5
        sz = size(sdf_vid);
        h5create(strcat(fpath,'.h5'), '/sdf_vid', sz, ...
            'FillValue', single(0.0), 'DataType', 'single', ...
            'ChunkSize', [1, sz(2), sz(3), sz(4)], 'Deflate', clevel)
        h5write(strcat(fpath,'.h5'), '/sdf_vid', sdf_vid)
        h5disp(strcat(fpath,'.h5'))  
        fprintf('Saved %s.h5\n', subdir_name);
    else
        save(fpath, 'sdf_vid');
        fprintf('Saved %s.mat\n', subdir_name);
    end
    
end
