clear
close all
% select NII file
[file,path]= uigetfile('BRATS/imagesTr/*.nii.gz','Select a File'); 
filename=cat(2,path,file);
label_file = ['BRATS/labelsTr/', file];

img = niftiread(filename);
label = niftiread(label_file);
segmentation = false(size(label));

%Iteration over all slices
for slice = 1:size(img, 3)

    img_slice_flair=mat2gray(img(:,:,slice,1));
    img_slice_t2w=mat2gray(img(:,:,slice,4));

    %thresholding
    T=0.5;
    B_flair=imbinarize(img_slice_flair,T);
    B_t2w=imbinarize(img_slice_t2w,T);

    %fill holes
    fill_flair = imfill(B_flair, "holes");
    fill_t2w = imfill(B_t2w, "holes");

    %erosion
    se = strel('disk', 1);
    eroded_flair = imerode(fill_flair, se);
    eroded_t2w = imerode(fill_t2w, se);

    segmentation(:, :, slice)=eroded_flair & eroded_t2w;
end

%find connected components
CC = bwconncomp(segmentation);

%dimension of every connected components
component_sizes = cellfun(@numel, CC.PixelIdxList);

%find index of largest component
[~, largest_component_idx] = max(component_sizes);

%binary mask
segmentation2 = false(size(segmentation));
segmentation2(CC.PixelIdxList{largest_component_idx}) = true;

%morphological operations
se2 = strel('sphere', 4 );
segmentation3 = imclose(segmentation2, se2);
segmentation4 = imopen(segmentation3, se2);
segmentation5 = imdilate(segmentation4, se2);

final_segmentation = img(:,:,:,1).*segmentation5;

%Dice Score
dice_score = dice(logical(final_segmentation), logical(label));
disp(['Dice Score: ' num2str(dice_score)]);

%Viewing a representative slice
slice_idx=77;
img_slice_flair=mat2gray(img(:,:,slice_idx,1));
img_slice_t2w=mat2gray(img(:,:,slice_idx,4));
T=0.5;
B_flair=imbinarize(img_slice_flair,T);
B_t2w=imbinarize(img_slice_t2w,T);
fill_flair = imfill(B_flair, "holes");
fill_t2w = imfill(B_t2w, "holes");
se = strel('disk', 1);
eroded_flair = imerode(fill_flair, se);
eroded_t2w = imerode(fill_t2w, se);

figure;
subplot(4, 4, 1);
imshow(mat2gray(img_slice_flair));
title('slice MRI Flair');
subplot(4, 4, 2);
imshow(B_flair);
title('thresholding');
subplot(4, 4, 3);
imshow(fill_flair);
title('fill');
subplot(4, 4, 4);
imshow(eroded_flair);
title('erosion');

subplot(4, 4, 5);
imshow(mat2gray(img_slice_t2w));
title('slice MRI t2w');
subplot(4, 4, 6);
imshow(B_t2w);
title('thresholding');
subplot(4, 4, 7);
imshow(fill_t2w);
title('fill');
subplot(4, 4, 8);
imshow(eroded_t2w);
title('erosion');

subplot(4, 4, 9);
imshow(segmentation(:, :, slice_idx));
title('flair & tw2');
subplot(4, 4, 10);
imshow(segmentation2(:, :, slice_idx));
title('largest component');
subplot(4, 4, 11);
imshow(segmentation3(:, :, slice_idx));
title('closing');
subplot(4, 4, 12);
imshow(segmentation4(:, :, slice_idx));
title('opening');

subplot(4, 4, 13);
imshow(segmentation5(:, :, slice_idx));
title('final segmentation');
subplot(4, 4, 14);
imshow(logical(label(:, :, slice_idx)));
title('label');

volumeViewer(final_segmentation);
volumeViewer(label);

%{
figure
isosurface(segmentation);
title('flair & t2w');
figure
isosurface(segmentation2);
title('largest component');
figure
isosurface(segmentation3);
title('closing');
figure
isosurface(segmentation4);
title('opening');
figure
isosurface(segmentation5);
title('dilation');
figure
isosurface(logical(label));
title('label');
%}
