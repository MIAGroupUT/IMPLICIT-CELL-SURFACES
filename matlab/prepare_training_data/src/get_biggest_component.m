function vol = get_biggest_component(vol)
    
    % determine connected components
    components = bwconncomp(vol);
    number_of_components = components.NumObjects;
    number_of_pixels = cellfun(@numel, components.PixelIdxList);
    [~, index_of_biggest_component] = max(number_of_pixels);

    % erase all but the biggest component
    for j = 1 : number_of_components

        % skip the biggest component
        if j == index_of_biggest_component
            continue;
        end

        % delete component i
        vol(components.PixelIdxList{j}) = 0;        
    end
end
