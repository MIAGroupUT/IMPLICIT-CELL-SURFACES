close all
clear
%clc

input_file = '../data/filo_cells/samples/filo_sdf_vid_demo_01.mat';
output_dir = './volume_filo_demo_01';

% desired output resolution
resolution = [256 256 256];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create output dir if it doesn't exist
if ~exist(output_dir, 'dir')
        mkdir(output_dir);
end

load(input_file);

for i = 1:size(sdf_vid,1)
    
    % get binary shape from SDF
    vol = squeeze(sdf_vid(i,:,:,:)) <= 0.035;
   
    % resize the volume
    centered_vol = imresize3(uint8(vol * 255), resolution, 'nearest');
    
    % save 3D volume
    output_file = strcat(output_dir, '/cell_filo_t', ...
                      sprintf('%03d', i-1),'.tif');   
    % process all z slices
    num_slices = size(vol, 3);
    imwrite(vol(:,:,1), output_file);
    for s = 2:num_slices
        imwrite(vol(:,:,s), output_file, 'WriteMode', 'append');
    end
    fprintf('Saved %s\n', output_file);
    
    % save preview (maximum intensity projection)
    output_file = strcat(output_dir, '/preview_cell_filo_t', ...
                      sprintf('%03d', i-1),'.png');
    imwrite(max(vol,[],3), output_file);
    
    fprintf('Saved %s\n', output_file);
    
end
