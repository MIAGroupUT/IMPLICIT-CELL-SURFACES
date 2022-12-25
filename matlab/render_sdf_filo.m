close all
clear
%clc

input_file = '../data/filo_cells/samples/filo_sdf_vid_demo_01.mat';
output_dir = './renders_filo_demo_01';

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
   
    % resize, filter & binarize the volume
    centered_vol = imresize3(uint8(vol * 255), resolution, 'nearest');
    centered_vol = imgaussfilt3(centered_vol, 2);

    % visualization
    isovalue = 192;
    isosurface(centered_vol, isovalue);
    axis([0 resolution(1)-1 0 resolution(2)-1 0 resolution(3)-1 0 255]);
    set(gca,'XTick',[0:resolution(1)/4:resolution(1)-1 resolution(1)-1]);
    xticklabels(num2cell(linspace(-1,1,5)))
    set(gca,'YTick',[0:resolution(2)/4:resolution(2)-1 resolution(2)-1]);
    yticklabels(num2cell(linspace(-1,1,5)))
    set(gca,'ZTick',[0:resolution(3)/4:resolution(3)-1 resolution(3)-1]);
    zticklabels(num2cell(linspace(-1,1,5)))
    set(gca,'FontSize',20);
    grid on;
    grid minor;
    colormap gray;
    set(gcf,'paperUnits','inches')
    set(gcf,'PaperSize', [8 8])
    set(gcf, 'PaperPosition',  [0.5, 0.5, 7, 7])
    set(gcf,'PaperPositionMode', 'Manual')
    
    % save image
    filename = strcat('cell_filo_t', sprintf('%03d', i-1));
    saveas(gcf,strcat(output_dir,'/',filename,'.png'));
    close;

end
