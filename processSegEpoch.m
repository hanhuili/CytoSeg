%% run an epoch for cytoplasm segmentation
function [net, info] = processSegEpoch(net, imgs, labels, is_train, learning_rate, use_gpu)
    img_num = size(imgs,3);
    sidx = randperm(img_num);
    img_num = numel(sidx);
    imgs = imgs(:,:,sidx);
    labels = labels(:,:,sidx);
    info = [];
    info.obj = 0;
    info.time = 0;
    % three segmentation losses
    seg_idx = net.getVarIndex('seg_loss');   
    so_idx = net.getVarIndex('so_loss');
    fo_idx = net.getVarIndex('fo_loss');
    net.vars(seg_idx).precious = true;
    net.vars(so_idx).precious = true;
    net.vars(fo_idx).precious = true;
    
    inputs{1} = 'input';
    inputs{3} = 'segLabel';
    inputs{5} = 'edgeLabel';
    
    if is_train
        pnum = numel(net.params);
        pmomentums = cell(pnum,1);
        for pi = 1 : pnum
            pmomentums{pi} = zeros(size(net.params(pi).value),'single');
            if use_gpu
                pmomentums{pi} = gpuArray(pmomentums{pi});
            end
        end
    end
    
    for si = 1 : img_num
        img = imgs(:,:,si);
        img = repmat(img, 1, 1, 3);
        img = imresize(img, net.meta.normalization.imageSize(1:2));
        label = labels(:,:,si);
        label = bwconvhull(label);
        label = imfill(label, 'holes');
        figure(1), imshow(label)
        label = imresize(label, net.meta.normalization.imageSize(1:2), 'nearest');
        elabel = bwmorph(label, 'remove');
        
        if is_train
            sz = net.meta.aug_scale(randsample(size(net.meta.aug_scale,1),1),:);
            sz = sz .* net.meta.normalization.imageSize(1:2);
            pos = net.meta.aug_trans(randsample(size(net.meta.aug_trans,1),1),:);
            pos = (pos + 0.5) .* net.meta.normalization.imageSize(1:2);
            theta = net.meta.aug_rotate(randsample(numel(net.meta.aug_rotate),1));
            img = getSubwindow(img, pos, sz, theta);
            label = getSubwindow(label, pos, sz, theta);
        end

        img = double(img) - net.meta.normalization.averageImage;
        img = single(img);
        label = single(label);
        elabel = single(elabel);
        if use_gpu
            img = gpuArray(img);
            label = gpuArray(label);
            elabel = gpuArray(elabel);
        end
        inputs{2} = img;
        inputs{4} = label;
        inputs{6} = elabel;
        tic();
        if is_train
            net.eval(inputs, {'so_loss', net.meta.so_weight, 'seg_loss', net.meta.seg_weight, 'fo_loss', net.meta.fo_weight});
            for pi = 1 : pnum
                thisLR = learning_rate * net.params(pi).learningRate;
                net.params(pi).value = net.params(pi).value  - thisLR * net.params(pi).der;
            end
        else
            net.eval(inputs);
        end
        info.time = info.time + toc();
        if is_train
            info.obj = info.obj + net.meta.so_weight * double(squeeze(gather(net.vars(so_idx).value))) / img_num;
            info.obj = info.obj + net.meta.seg_weight * double(squeeze(gather(net.vars(seg_idx).value))) / img_num;
            info.obj = info.obj + net.meta.fo_weight * double(squeeze(gather(net.vars(fo_idx).value))) / img_num;
        else
            pred = net.vars(net.getVarIndex('fuse_out')).value >= 0.5;
            pred = pred .* net.meta.error_mask;
            info.obj = info.obj + getIOU(pred, label) / img_num;
        end
    end
end


function patch = getSubwindow(im, pos, sz, theta)
    im_sz = size(im);
    theta = theta * pi / 180;
    cos_theta = cos(theta);
    sin_theta = sin(theta);
    xidx = floor(pos(2) - sz(2)/2);
    xidx = xidx : xidx + sz(2) - 1;
    yidx = floor(pos(1) - sz(1)/2);
    yidx = yidx : yidx + sz(1) - 1;
    [rxs,rys] = meshgrid(xidx,yidx);
    rxs = rxs - pos(2);
    rys = pos(1) - rys;
	xs = rxs * cos_theta - rys * sin_theta;
    ys = rys * cos_theta + rxs * sin_theta;
    xs = xs + pos(2);
    ys = pos(1) - ys;
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);
	ys(ys > size(im,1)) = size(im,1);
    ys = round(ys);
    xs = round(xs);
	%extract image
    idx = sub2ind([size(im,1), size(im,2)], ys, xs);
    if size(im,3) == 3
        im1 = im(:,:,1);
        im1 = im1(idx);
        im2 = im(:,:,2);
        im2 = im2(idx);
        im3 = im(:,:,3);
        im3 = im3(idx);
        patch = cat(3,im1,im2,im3);
    else
        patch = im(idx);
    end
    patch = imresize(patch,[im_sz(1:2)]);
end

