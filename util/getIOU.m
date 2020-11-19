%% get the IOU between two binary images
function score = getIOU(pd, gt)
    its = pd > 0 & gt > 0;
    uin = pd > 0 | gt > 0;
    score = sum(its(:) > 0) / sum(uin(:) > 0);
end

