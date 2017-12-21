function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
% 特征标准化会返回每个特征的平均值是0,标准差为1的特征.
% mu表示平均自,sigma表示标准差
% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. 
% 首先,对于每一个特征,计算特征的平均值,每个数据都减去平均值,并且将平均值存储为mu
% Next, compute the standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
% 然后,对每一特征计算标准差,使用新的特征值处以标准差,并且把标准差记为sigma 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%  注意对于数组X,每一个列是一个特征,每一行是一组训练数据.
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
