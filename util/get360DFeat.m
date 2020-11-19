%% get the 360D distances as the feature for irregular cell recogntion, input img is the binary mask
function feat = get360DFeat(img)
    img = refineMask(img);
    centerR = size(img,1)/2;
    centerC = size(img,2)/2;
    feat = zeros(360,1);
    [idxR,idxC] = find(img);
    [theta,rho] = mycart2pol(idxC - centerC, centerR - idxR);
    mrho = zeros(360,1);
    for mi = 1 : 360
        tidx = find(theta == mi);
        if numel(tidx) > 0
            mrho(mi) = mean(rho(tidx));
        end
    end
    mrho = fillList(mrho,inf);
    feat(:,1) = (mrho - mean(mrho)).^2 / mean(mrho);
end


function [theta,rho] = mycart2pol(x,y)
    [theta,rho] = cart2pol(x,y);
    theta = round(theta * 180 / pi);
    theta(theta < 0) = theta(theta < 0) + 360;
    theta(theta == 0) = 360;
end

function [list, findMaxGap] = fillList(list,maxGap)
    findMaxGap = 0;
    listLength = numel(list);
    for ti = 1 : listLength
        if list(ti) == 0
            idx = ti;
            front = ti - 1;
            if front == 0 front = listLength;end
            while(list(front)==0)
                idx = [idx;front];
                front = front - 1;
                if front == 0 front = listLength;end
            end
            tail = ti + 1;
            if tail == listLength + 1 tail = 1; end
            while(list(tail)==0)
                idx = [idx;tail];
                tail = tail + 1;
                if tail == listLength + 1 tail = 1; end
            end
            if numel(idx) <= maxGap
                interv = (list(tail) - list(front)) / (numel(idx) + 1);
                list(idx) = list(front) + [1 : numel(idx)] * interv;
            end
            if numel(idx) > findMaxGap
                findMaxGap = numel(idx);
            end
        end
    end
end