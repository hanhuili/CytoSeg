%% The architecture of the irregular zygote recognition network
function net = initZRNetwork(positive_weight, average_image)
    net = dagnn.DagNN();
    net.addLayer('conv1Layer',dagnn.Conv('size',[360,1,1,128]),{'input'},{'conv1_act'},{'conv1_filter','conv1_bias'});
    net.addLayer('act1Layer',dagnn.Sigmoid(),{'conv1_act'},{'conv1_out'});
    net.addLayer('conv2Layer',dagnn.Conv('size',[1,1,128,1]),{'conv1_out'},{'conv2_act'},{'conv2_filter','conv2_bias'});
    net.addLayer('act2Layer',dagnn.Sigmoid(),{'conv2_act'},{'probOut'});
    net.addLayer('LossLayer',CELoss('positiveWeight', positive_weight),{'probOut','label'},{'error'})
    
    net.initParams();    
    net.meta.batch_size = 16;
    net.meta.learning_rate = logspace(-2,-4,100);
    net.meta.normalization.average_image = average_image;
    net.conserveMemory = false;
end


