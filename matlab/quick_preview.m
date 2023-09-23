close all
clear
%clc

%input_dir = './OUT_randomgen';
%output_dir = './OUT_randomgen_preview';
input_dir = './OUT_reconstruct';
output_dir = './OUT_reconstruct_preview';

use_HDF5 = true; % set input to MAT or HDF5 files

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create output dir if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% list of input files
if use_HDF5
    input_list = dir(strcat(input_dir, '/*.h5'));
else
    input_list = dir(strcat(input_dir, '/*.mat'));
end

for i = 1:length(input_list)
    
    input_file = strcat(input_dir, '/', input_list(i).name);
    
    if use_HDF5
        h5disp(input_file)
        sdf_vid = h5read(input_file, '/sdf_vid');
    else
        load(input_file);
    end
    
    for frame = 1:size(sdf_vid,1)
        
        input_volume = squeeze(sdf_vid(frame,:,:,:)) <= 0.0;
        
        img = uint8(max(input_volume,[],3)) * 255;
    
        [~, filename, ~] = fileparts(input_file);

        imwrite(img, ...
                strcat(output_dir, '/', filename, '_t', ...
                sprintf('%03d',frame-1), '.png'));
        
    end
    
end
