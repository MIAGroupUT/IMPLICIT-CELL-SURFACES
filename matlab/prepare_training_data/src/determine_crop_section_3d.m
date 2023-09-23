% determine crop section (center the organoid according to its centroid)
function [rows, columns, slices] = determine_crop_section_3d(mask, cropped_size)

    % determine size of rotated image
	tmp_size = size(mask);

	% compute centroid
	centroid = regionprops(mask, 'centroid');
	centroid = round(centroid.Centroid);
	centroid = [centroid(2) centroid(1) centroid(3)]; % x,y -> y,x (rows, columns)

	% compute crop region boundary (inclusive)
	step = cropped_size / 2;
	crop_boundary = struct('rows', [0 0], 'columns', [0 0], 'slices', [0 0]);
	% rows
	crop_boundary.rows(1) = centroid(1) - step(1);
	crop_boundary.rows(2) = centroid(1) + step(1) - 1;
	if crop_boundary.rows(1) < 1
		crop_boundary.rows(2) = crop_boundary.rows(2) + abs(crop_boundary.rows(1)) + 1;
		crop_boundary.rows(1) = 1;
	elseif crop_boundary.rows(2) > tmp_size(1)
		crop_boundary.rows(1) = crop_boundary.rows(1) - (crop_boundary.rows(2) - tmp_size(1));
		crop_boundary.rows(2) = tmp_size(1);
	end 
	% columns
	crop_boundary.columns(1) = centroid(2) - step(2);
	crop_boundary.columns(2) = centroid(2) + step(2) - 1;
	if crop_boundary.columns(1) < 1
		crop_boundary.columns(2) = crop_boundary.columns(2) + abs(crop_boundary.columns(1)) + 1;
		crop_boundary.columns(1) = 1;
	elseif crop_boundary.columns(2) > tmp_size(2)
		crop_boundary.columns(1) = crop_boundary.columns(1) - (crop_boundary.columns(2) - tmp_size(2));
		crop_boundary.columns(2) = tmp_size(2);
    end 
    % slices
	crop_boundary.slices(1) = centroid(3) - step(3);
	crop_boundary.slices(2) = centroid(3) + step(3) - 1;
	if crop_boundary.slices(1) < 1
		crop_boundary.slices(2) = crop_boundary.slices(2) + abs(crop_boundary.slices(1)) + 1;
		crop_boundary.slices(1) = 1;
	elseif crop_boundary.slices(2) > tmp_size(2)
		crop_boundary.slices(1) = crop_boundary.slices(1) - (crop_boundary.slices(2) - tmp_size(2));
		crop_boundary.slices(2) = tmp_size(2);
	end 
	
	rows = crop_boundary.rows(1):crop_boundary.rows(2);
	columns = crop_boundary.columns(1):crop_boundary.columns(2);
    slices = crop_boundary.slices(1):crop_boundary.slices(2); 
	
end