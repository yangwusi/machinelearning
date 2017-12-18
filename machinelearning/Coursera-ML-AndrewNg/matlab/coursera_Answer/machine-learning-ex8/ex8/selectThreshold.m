function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
    %离散数学分类方法，p(x)>epsion ,prediction==1，反之prediction==0
    predictions = (pval < epsilon);
    tp = sum((predictions == 1) & (yval == 1));
    fp = sum((predictions == 1) & (yval == 0));
    fn = sum((predictions == 0) & (yval == 1));
    tn = sum((predictions == 0) & (yval == 0));
    
    %求解prec
     if tp + fp ~= 0
        prec = tp / (tp + fp);
    else
        prec = 0;
     end
    
    %求解rec
    if tp + fn ~= 0
        rec = tp / (tp + fn);
    else
        rec = 0;
    end
    
    %求解F1
    if prec + rec ~= 0
        F1 = 2 * prec * rec / (prec + rec);     %基于查准率与查全率的调和平均
    else
        F1 = 0;
    end
    

    % =============================================================
    %选出最合适的epsilon   F1越大，epsilon越准确
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
