%% demo for zygote recognition
close all;
clear;
use_gpu = false;

%% These 360D features can be obtained via the get360DFeat.m function
%load('360DFeats.mat');  % feats and labels

result_path = 'results\';
if ~exist(result_path,'dir') mkdir(result_path); end
for si = 1 : size(splits,1)     % cross-validation splits
    train_idxs = splits{si,1};
    test_idxs = splits{si,2};
    positive_weight = 1 - sum(labels(train_idxs)) / numel(train_idxs);
    net = initZRNetwork(positive_weight, mean(feats(:,:,train_idxs),3));    
    if use_gpu
        net.move('gpu');
    end
    epoch_num = numel(net.meta.learning_rate);
    train_obj = zeros(1,epoch_num);
    test_obj = zeros(1,epoch_num);
    for ni = 1 : epoch_num
        lr = net.meta.learning_rate(ni);
        [net, info] = processZREpoch(net, feats(:,:,train_idxs), labels(train_idxs), true, lr, use_gpu);
        train_obj(ni) = info.obj;
        fprintf('epoch %d training, object function value %f, consumed %f seconds\n',ni, info.obj, info.time);
        [net, info,results] = processZREpoch(net, feats(:,:,test_idxs), labels(test_idxs), false, lr, use_gpu);
        fprintf('epoch %d test, object function value %f, consumed %f seconds\n',ni, info.obj, info.time);
        test_obj(ni) = info.obj;
        
        figure(1);
        hold on;
        plot([1:ni],train_obj(1:ni));
        plot([1:ni],test_obj(1:ni));
        hold off;
        drawnow;
    end
end
end