%% Cross entropy loss for classification
classdef CELoss < dagnn.ElementWise
    properties
        sv = 1e-6;
        positiveWeight = 0.5;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            preds = squeeze(inputs{1});
            labels = squeeze(inputs{2});
            preds(preds < obj.sv) = obj.sv;
            preds(preds > (1 - obj.sv)) = 1 - obj.sv;
            crLoss = - obj.positiveWeight * log(preds) .* labels - (1 - obj.positiveWeight) * log(1 - preds) .* (1 - labels);
            outputs{1} = sum(crLoss(:));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            labels = squeeze(inputs{2});
            Y = squeeze(inputs{1});
            Y(Y < obj.sv) = obj.sv;
            Y(Y > (1 - obj.sv)) = 1 - obj.sv;
            Y = - obj.positiveWeight * (1 ./ Y) .* labels + (1 - obj.positiveWeight) * (1 ./ (1-Y)) .* (1 - labels);
            derInputs{1} = reshape(Y, size(inputs{1}))* derOutputs{1};
            derParams = {} ;
            derInputs{2} = [];
        end
        
        function obj = CELoss(varargin)
            obj.load(varargin);
        end
    end
end

