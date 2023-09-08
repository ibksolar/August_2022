
base_dir = 'Y:\ibikunle\Python_Project\Fall_2021\Predictions_Folder\DL_models_predictions_folder_final';
sub_dir = {'L1','L2','L3'};

% mod_names = cell(1,7);
% mod_names{1} = 'all_AttUNet';
% mod_names{2} = 'all_DeepLab';
% mod_names{3} = 'all_ensemble';
% mod_names{4} = 'all_FCN';
% mod_names{5} = 'all_SimpleUNet';

for sub_idx = 1:length(sub_dir)
    dir_path = fullfile(base_dir,sub_dir{sub_idx});
    
    all_dirs = dir(dir_path);
    dirFlag = [all_dirs.isdir];
    subDirs = all_dirs(dirFlag); % A structure with extra info.
    model_dir = {subDirs(3:end).name};
    
    for mod_idx = 1:length(model_dir)
        if mod_idx ~= 3 && mod_idx ~= 8 % Skip ensemble2 and Deeplab2
            all_files = dir( fullfile(base_dir,sub_dir{sub_idx},model_dir{mod_idx},'*.mat') );

            for file_idx = 1:length(all_files)
                curr_path =  fullfile(base_dir,sub_dir{sub_idx},model_dir{mod_idx},all_files(file_idx).name);
                matObj = load(curr_path);
                x = matObj.model_output;
                E=convTri(single(x),1);
                [Ox,Oy]=gradient2(convTri(E,4));
                [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
                O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
                nms_out2=edgesNmsMex(E,O,1,5,1.01,4);
                %             imwrite(uint8(E*255),[root_res, res_names{file_idx}])
                save(curr_path,'-append','nms_out2')
            end
        end
        
    end
end
