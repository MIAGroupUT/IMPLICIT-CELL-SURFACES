close all
clear
%clc

input_dir = './reconstructions';

output_dir = './reconstruction_preview';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create output dir if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% list of input files
input_list = dir(strcat(input_dir, '/*.mat'));

for i = 1:length(input_list)

    input_file = strcat(input_dir, '/', input_list(i).name);
    
    load(input_file);
    
    [~, filename, ~] = fileparts(input_file);
    
    imwrite(uint8(sdf_norm <= 0) * 255, ...
            strcat(output_dir, '/', filename, '.png'));
    
end
