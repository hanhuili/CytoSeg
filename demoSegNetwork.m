addpath('util');
clear;
close all;
load('data.mat');  % dataset for the NYU mouse embryo images

%% configurations
cv_num = 5;     % k-fold cross validation
use_gpu = false;
split_num = ceil(numel(subjects) / cv_num);
imgh = 256;
imgw = 256;
resultPath = 'result';
if ~exist(resultPath, 'dir') 
    mkdir(resultPath); 
end

%% cross-validation
for cvi = 1 : cv_num
    % gather images and labels
    split_sidx = ((cvi - 1) * split_num) + 1;
    split_eidx = min([cvi * split_num, numel(subjects)]);
    test_imgs = [];
    test_labels = [];
    for si = split_sidx : split_eidx
       subject = subjects{si};
       for sii = 1 : numel(subject)
           if isempty(test_imgs)
               test_imgs = subject{sii}{1};
               test_labels = subject{sii}{2};
           else
               test_imgs(:, :, end + 1) = subject{sii}{1};
               test_labels(:, :, end + 1) = subject{sii}{2};
           end
       end
    end
    train_subject_idxs = setdiff([1 : numel(subjects)], [split_sidx : split_eidx]);
    train_imgs = [];
    train_labels = [];
    for si = 1 : numel(train_subject_idxs)
       subject = subjects{train_subject_idxs(si)};
       for sii = 1 : numel(subject)
           if isempty(train_imgs)
               train_imgs = subject{sii}{1};
               train_labels = subject{sii}{2};
           else
               train_imgs(:, :, end + 1) = subject{sii}{1};
               train_labels(:, :, end + 1) = subject{sii}{2};
           end
       end
    end
    
    % initialize network
    net = initSegNetwork(imgh, imgw);
    if use_gpu
        net.move('gpu');
    end
    epoch_info = [];
    epoch_num = numel(net.meta.learning_rate);
    train_obj = zeros(1, epoch_num);
    test_obj = zeros(1, epoch_num);
    
    % training and test
    for ni = 1 : epoch_num
        lr = net.meta.learning_rate(ni);
        
        % training
        [net, info] = processSegEpoch(net, train_imgs, train_labels, true, lr, use_gpu);
        epoch_info(end+1).trainInfo = info;
        train_obj(ni) = info.obj;
        fprintf('epoch %d training, objective function value %f, consumed %f seconds\n',ni, info.obj, info.time);
        
        % test
        [net, info] = processSegEpoch(net, test_imgs, test_labels, false, lr, use_gpu);
        fprintf('epoch %d test, objective function value %f, consumed %f seconds\n',ni, info.obj, info.time);
        epoch_info(end).testInfo = info;
        test_obj(ni) = info.obj;
        
        % draw objective function value
        figure(1);
        hold on;
        plot([1:ni],train_obj(1:ni));
        plot([1:ni],test_obj(1:ni));
        hold off;
        drawnow;
        
        % randomly select an image for demonstration
        figure(2);
        img = test_imgs(:,:,(randsample(size(test_labels, 3),1)));
        img = imresize(img, [imgh, imgw]);
        img = repmat(img, 1, 1, 3);
        imshow(img);
        figure(3);
        
        img = double(img) - net.meta.normalization.averageImage;
        img = single(img);
        if use_gpu
            img = gpuArray(img);
        end
        net.eval({'input', img});
        
        imshow(net.vars(net.getVarIndex('fuse_out')).value >= 0.5);
        % alternatively, use the refineMask function to get smoother result
        % before showing
 
        
        % save the model for every # epochs
        if mod(ni,5) == 0
            net.move('cpu');
            net_copy = net.copy();
            if use_gpu
                net.move('gpu');
            end
            save(fullfile(resultPath, ['cv', num2str(cvi), '-', num2str(ni),'.mat']), 'net_copy','epoch_info', '-v7.3');
        end
    end
end



