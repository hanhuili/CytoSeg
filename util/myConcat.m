%% Wrapper for the concatenation function
classdef myConcat < dagnn.ElementWise
    properties
        dim = 3;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = cat(3,inputs{:});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            inputNum = numel(inputs);
            derInputs = cell(1,inputNum);
            for ni = 1 : inputNum
                derInputs{ni} = derOutputs{1}(:,:,ni);
            end
            derParams = {} ;
        end
    end
end

