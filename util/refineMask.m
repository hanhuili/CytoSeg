%% refine the mask, remove isolated points
function filted_img = refineMask(img)
    img = imdilate(img, strel('disk',1));
    img = imfill(img, 'holes');
    img = bwareafilt(img,1);
    img = bwconvhull(img);
    filted_img = imerode(img, strel('disk',1)); 
end
