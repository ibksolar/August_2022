%create_3D_raster for regression

base_aug_path = 'Y:\ibikunle\Python_Project\Fall_2021\all_block_data\Attention_Train_data\Full_size_data\test_data';


all_files = get_filenames(base_aug_path,'20','_0','.mat');

for iter = 1:length(all_files)
    curr_file = load(all_files{iter});
    
    [Nt,Nx] = size(curr_file.echo_tmp);

    vec_layer = round(nan_fir_dec(curr_file.vec_layer,ones(1,11)/11,1) );

    raster_3D = zeros(Nt,Nx,3);

    for rw_idx = 1:size(vec_layer,1)
        if ~all(isnan(vec_layer(rw_idx,:)))

            for col_idx = 1:size(vec_layer,2)
                if ~isnan(vec_layer(rw_idx,col_idx)) 
                    raster_rw = vec_layer(rw_idx,col_idx);
                    raster_3D(raster_rw,col_idx,:) = [1,rw_idx,raster_rw];
                end
            end
        end
    end
    
    raster_3D(:,:,3) = raster_3D(:,:,3)/Nt; % Normalize

    save(all_files{iter},"raster_3D",'-append');

end

   

