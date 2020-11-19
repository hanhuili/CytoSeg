%% Segmentation loss
classdef segLoss < dagnn.ElementWise
    properties
        sv = 1e-6;
        errorMask;
        beta = 0.8;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            layerNum = numel(inputs);
            label = inputs{layerNum};
            pPtMask = single(label);
            if isa(label, 'gpuArray')
                pPtMask = gpuArray(pPtMask);
            end
            nPtMask = 1 - pPtMask;
            beta = obj.beta;
%             beta = sum(nPtMask(:)) / numel(label);
            nbeta = 1 - beta;
            crLoss = 0;
            for li = 1 : layerNum - 1
                tlabel = inputs{li};
                tlabel(tlabel < obj.sv) = obj.sv;
                tlabel(tlabel > (1 - obj.sv)) = 1 - obj.sv;
                tempLoss = - beta * log(tlabel) .* pPtMask - nbeta * log(1 - tlabel) .* nPtMask;
                tempLoss = tempLoss .* obj.errorMask;
                crLoss = crLoss + sum(tempLoss(:));
            end
            outputs{1} = crLoss;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            layerNum = numel(inputs);
            label = inputs{layerNum};
            pPtMask = single(label);
            if isa(label, 'gpuArray')
                pPtMask = gpuArray(pPtMask);
            end
            nPtMask = 1 - pPtMask;
            beta = obj.beta;
%             beta = sum(nPtMask(:)) / numel(label);
            nbeta = 1 - beta;
            lossWeight = derOutputs{1};
            derInputs = cell(1,layerNum);
            for li = 1 : layerNum - 1
                Y = inputs{li};
                Y(Y < obj.sv) = obj.sv;
                Y(Y > (1 - obj.sv)) = 1 - obj.sv;
                Y = - beta * (1 ./ Y) .* pPtMask + nbeta * (1 ./ (1-Y)) .* nPtMask;
                Y = Y .* obj.errorMask;
                derInputs{li} = Y * lossWeight * (1 / (layerNum - 1));
            end
            derParams = {} ;
        end
        
        function obj = segLoss(varargin)
            obj.load(varargin) ;
        end
    end
end

