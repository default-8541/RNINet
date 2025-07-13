file_path = ''; % input HQ path
save_path = '';  % output noisy path
mkdir(save_path);

%img_path_list = dir(strcat(file_path,'*.png'));  
img_path_list1 = dir(strcat(file_path,'*.png')); 
img_path_list2 = dir(strcat(file_path,'*.bmp')); 
img_path_list3 = dir(strcat(file_path,'*.mat')); 
img_path_list = [img_path_list1;img_path_list2;img_path_list3];

img_num = length(img_path_list);   
fprintf('向量的长度是：%d\n', img_num);
if img_num > 0   
    for j = 1:img_num   
        image_name = img_path_list(j).name;   
        I =  imread(strcat(file_path,image_name));
        fprintf('%d %s\n', j, strcat(file_path,image_name)); 

        path = strcat(save_path, image_name);
        
        % speckle
        speckle0024 = imnoise(I,'speckle', 0.014); 
        
        % salt & pepper
        %sp0002 = imnoise(I, 'salt & pepper', 0.004);   

        imwrite(speckle0024, path);

    end
end