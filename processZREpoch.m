function [net, info] = processZREpoch(net, imgs, labels, is_train, learning_rate, use_gpu)
    img_num = numel(labels);
    sidx = randperm(img_num);
    imgs = imgs(:,:,sidx);
    labels = labels(sidx);
    info = [];
    info.obj = 0;
    info.time = 0;
    eIdx = net.getVarIndex('probOut');
    net.vars(eIdx).precious = true;
    inputs{1} = 'input';
    inputs{3} = 'label';
    batch_size = net.meta.batch_size;
    for bi = 1 : ceil(img_num / batch_size)
        sidx = (bi-1) * batch_size + 1;
        eidx = min([bi*batch_size, img_num]);
        this_batch_size = eidx - sidx + 1;
        batch_imgs = imgs(:,:,sidx:eidx);
        batch_labels = labels(sidx:eidx);
        batch_imgs = reshape(batch_imgs,size(batch_imgs,1), size(batch_imgs,2), 1, size(batch_imgs,3));
        batch_imgs = bsxfun(@minus,batch_imgs,net.meta.normalization.average_image);
        inputs{2} = single(batch_imgs);
        inputs{4} = single(batch_labels);
        if use_gpu
            inputs{2} = gpuArray(inputs{2});
            inputs{4} = gpuArray(inputs{4});
        end
        tic();
        if is_train
            net.eval(inputs, {'error',1});
            for pi = 1 : numel(net.params)
                thisLR = learning_rate * net.params(pi).learningRate / this_batch_size;
                net.params(pi).value = net.params(pi).value  - thisLR * net.params(pi).der;
            end
        else
            net.eval(inputs);
        end
        info.time = info.time + toc();
        preds = double(squeeze(gather(net.vars(eIdx).value)));
        preds(preds < 0.5) = 0;
        preds(preds ~= 0) = 1;
        info.obj = info.obj +  sum(preds(:) == batch_labels(:));
    end
    info.obj = info.obj / img_num;
end




