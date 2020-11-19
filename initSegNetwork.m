%% initialize the segmentation network
function net = initSegNetwork(imgh, imgw)
    pretrain_path = 'imagenet-vgg-verydeep-16.mat';
    load(pretrain_path);
    net = dagnn.DagNN.fromSimpleNN(net, 'CanonicalNames', true);
    while(~strcmp(net.layers(end).name,'relu5_3')) net.removeLayer(net.layers(end).name); end  % remove layers behind relu5_3
    net.meta.normalization.averageImage = imresize(net.meta.normalization.averageImage, [imgh, imgw]);
    net.meta.normalization.imageSize = [imgh, imgw, 3];
    
    % ignore the boundary of img
    error_pad = 10;
    error_mask = ones(imgh, imgw, 'single');
    error_mask(1:error_pad, :) = 0;
    error_mask(:, 1:error_pad) = 0;
    error_mask(end - error_pad + 1 : end,:) = 0;
    error_mask(:,end - error_pad + 1 : end) = 0;
    net.meta.error_pad = error_pad;
    net.meta.error_mask = error_mask;
    
    %% side output layers
    so_layer_names = {'conv2_2', 'conv3_3'};
    so_scale_factors = [2, 4];  
    so_num = numel(so_layer_names);
    so_name = cell(1,so_num);
    for si = 1 : so_num
        [net, so_name{si}]= addSideOutLayer(net,so_layer_names{si},so_scale_factors(si));
    end
    
    net.addLayer('sideOutConcatLayer',myConcat(),so_name,{'so_concat'});
    net.addLayer('sideOutFuseLayer',dagnn.Conv('size',[1,1,so_num,1]),{'so_concat'},{'so_act'},{'so_filter','so_bias'});
    net.addLayer('sideOutActLayer',dagnn.Sigmoid(),{'so_act'},{'so_out'});
    net.params(net.getParamIndex('so_filter')).value = randn(1,1,so_num,1,'single') * sqrt(2 / so_num);
    net.params(net.getParamIndex('so_bias')).value = zeros(1,1,'single');
    net.addLayer('sideOutLossLayer',segLoss('errorMask',error_mask),{'so_out','edgeLabel'},{'so_loss'});
    
    %% main branch
    seg_scale_factor = 16;  
    feat_dim = 512;
    seg_out_name = net.layers(net.getLayerIndex('relu5_3')).outputs{1};
    net.addLayer('segDeconvLayer',dagnn.ConvTranspose('size',[seg_scale_factor,seg_scale_factor,1,feat_dim],'upsample',[seg_scale_factor,seg_scale_factor]),{seg_out_name},{'seg_act'},{'seg_dfilter','seg_dbias'});
    net.addLayer('segActLayer',dagnn.Sigmoid(),{'seg_act'},{'seg_out'});
    net.addLayer('segLossLayer',segLoss('errorMask',error_mask),{'seg_out','segLabel'},{'seg_loss'});
    sc = sqrt(2 / (seg_scale_factor * seg_scale_factor * feat_dim));
    net.params(net.getParamIndex('seg_dfilter')).value = randn(seg_scale_factor,seg_scale_factor,1,feat_dim,'single') * sc;
    net.params(net.getParamIndex('seg_dbias')).value = zeros(1,1,'single');
    
    %% Extra layer for fusing the outputs of the main branch and side branches
    net.addLayer('fuseOutConcatLayer',myConcat(),{'so_out','seg_out'},{'fo_in'});
    net.addLayer('fuseOutConvLayer',dagnn.Conv('size',[1,1,2,1]),{'fo_in'},{'fo_act'},{'fo_filter','fo_bias'});
    net.addLayer('fuseOutActLayer',dagnn.Sigmoid(),{'fo_act'},{'fuse_out'});
    net.addLayer('fuseOutLossLayer',segLoss('errorMask',error_mask),{'fuse_out','segLabel'},{'fo_loss'});
    net.params(net.getParamIndex('fo_filter')).value = randn(1,1,2,1,'single') * sqrt(2 / 2);
    net.params(net.getParamIndex('fo_bias')).value = zeros(1,1,'single');
    
    %% Training related parameters
    [aug_scale_h,aug_scale_w] = meshgrid([0.8:0.05:1.2]);
    [aug_trans_h,aug_trans_w] = meshgrid([-0.10:0.02:0.10]);
    net.meta.aug_scale = [aug_scale_h(:),aug_scale_w(:)];
    net.meta.aug_rotate = [0:359];
    net.meta.aug_trans = [aug_trans_h(:),aug_trans_w(:)];
    net.meta.learning_rate = logspace(-6, -8, 50);
    net.meta.so_weight = 0.5;
    net.meta.seg_weight = 1;
    net.meta.fo_weight = 1;
    net.meta.momentum = 0.9;
    net.meta.weight_decay = 5 * 1e-6;
    net.conserveMemory = false;
end



function [net, side_out_name, feat_dim]= addSideOutLayer(net, layer_name, scale_factor)
    lidx = net.getLayerIndex(layer_name);
    feat_dim = net.layers(lidx).block.size(4);
    output_name = net.layers(lidx).outputs{1};
    net.addLayer([layer_name,'_edgeLayer'], dagnn.Conv('size',[1,1,feat_dim,1]), {output_name}, {[output_name,'_conv_out']}, {[layer_name,'_efilter'], [layer_name,'_ebias']});
    net.params(net.getParamIndex([layer_name,'_efilter'])).value = randn(1,1,feat_dim,1,'single') * sqrt(2 / (feat_dim));
    net.params(net.getParamIndex([layer_name,'_ebias'])).value = zeros(1,1,'single');
    net.addLayer([layer_name,'_dc_act'],dagnn.Sigmoid(),{[output_name,'_conv_out']},{[output_name,'_sa']});
    side_out_name = [layer_name,'_so'];
    net.addLayer([layer_name,'_deconv'],dagnn.ConvTranspose('size',[scale_factor,scale_factor,1,1],'upsample',[scale_factor,scale_factor]),{[output_name,'_sa']},{side_out_name},{[layer_name,'_dfilter'],[layer_name,'_dbias']});
    net.params(net.getParamIndex([layer_name,'_dfilter'])).value = randn([scale_factor,scale_factor,1,1],'single') * sqrt(2 / prod([scale_factor,scale_factor,1,1]));
    net.params(net.getParamIndex([layer_name,'_dbias'])).value = zeros(1,1,'single');
end
