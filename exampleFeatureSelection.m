function [features,data] = exampleFeatureSelection(data,output,c_thresh,...
    type,n_cycles,rf_thresh,lambda,l_thresh)
% This program gives an example of feature selection. It removes
% duplicates, uses Pearson's linear correlation coefficient, random forest,
% and L1 regularization for classification and regression.
%
% Inputs:
% data - the original dataset of features
% output - the labels associated with the dataset
% c_thresh - the desired threshold for the correlation coefficient features
% type - classification (1) or regression (0)
% n_cycles - number of cycles for the random forest
% rf_thresh - the desired threshold for the random forest features
% lambda - lambda value chosen for the L1 regularization
% l_thresh - the desired threshold for the L1 regularization features
%
% Outputs:
% features - this gives a list of the final features selected (noted by 1)
% data - this is the final dataset with the features selected

% Remove duplicate columns
features = ones(size(data,2),1);
tmp = data;
data = unique(data','rows','stable')';
n_feat = size(data,2);

% Performs Pearson's linear correlation coefficient & removes the features 
% over the determined threshold
corr_mat = corr(data);
corr_mat = corr_mat - diag(diag(corr_mat));
[corrX, corrY] = find(corr_mat>c_thresh);
for i=1:length(corrX)
    idx = find(corrY == corrX(i));
    corrX(idx,:) = 0;
    corrY(idx,:) = 0;
end
corrX = unique(corrX);
corrX = corrX(2:end);
c_feat = setxor(corrX,(1:n_feat)');
data = data(:,c_feat);
% features(c_feat,2) = 1;

% Trains a random forest of classification or regression trees and then 
% estimates predictor importance for each feature
if type
    Learner = 'svm';
    Mdl = fitcensemble(data,output,'Method','Bag','NumLearningCycles',n_cycles);
else
    Learner = 'logistic';
    Mdl = fitrensemble(data,output,'Method','Bag','NumLearningCycles',n_cycles);
end
imp = oobPermutedPredictorImportance(Mdl);
figure;
bar(imp);
title('Out-of-Bag Permuted Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

r_feat = imp>rf_thresh;
data = data(:,r_feat);


%Use L1 Regularization to prune the features
t = templateLinear('Learner',Learner,'Regularization','lasso','lambda',lambda);
Mdl2 = fitcecoc(data,output,'Learners',t);
l_feat = Mdl2.BinaryLearners{1,1}.Beta>l_thresh;
data = data(:,l_feat);

[~,ia] = setdiff(tmp',data','rows');
features(ia) = 0;

