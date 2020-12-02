function JakobVersioncovid19Detection_SVM_densenet201()

%% *Covid Detection Transfer Learning (densenet201)*
%% Load Data 

% Random data split
% imds = imageDatastore(fullfile('images_and_split',{'CT_COVID','CT_NonCOVID'}),'IncludeSubfolders', ...
%      true,'LabelSource','foldernames');

% Test different splits of training, validation, and testing.
% Make sure percentages add up to 1.

% % Training Split
% percent_train = 0.75;
% % Validation Split
% percent_val = 0.125;
% % Testing Split
% percent_test = 1 - (percent_train + percent_val);
% 
% % Check for invalid percentage.
% if(percent_test < 0 || percent_val < 0 || percent_train < 0 || ...
%    percent_test > 1 || percent_val > 1 || percent_train > 1)
%     fprintf("Invalid percentages.")
%     return;
% end
% 
% [imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,percent_train,percent_val,percent_test,'randomize');
%% Load Pretrained Network
% 

imdsTest = imageDatastore(fullfile('ucsd_Data','test'),'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain = imageDatastore(fullfile('ucsd_Data',{'train','validation'}),'IncludeSubfolders',true,'LabelSource','foldernames');

net = densenet201;
%% 
% Use |analyzeNetwork| to display an interactive visualization of the network 
% architecture and detailed information about the network layers.
% analyzeNetwork(net)


%% 
% 
%% 
% 


net.Layers(1)
inputSize = net.Layers(1).InputSize;



%Extract Image Features


%  pixelRange = [-30 30];
% scaleRange = [0.9 1.1];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...
%     'RandXScale',scaleRange, ...
%     'RandYScale',scaleRange);
% augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
%     'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
% 
% augimdsTest = augmentedImageDatastore(inputSize,imdsTest,"ColorPreprocessing","gray2rgb");
%  pixelRange = [-30 30];
% scaleRange = [0.9 1.1];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...
%     'RandXScale',scaleRange, ...
%     'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,"ColorPreprocessing","gray2rgb");
augimdsTest = augmentedImageDatastore(inputSize,imdsTest,"ColorPreprocessing","gray2rgb");


% *SVM CLASSIFIER*
%deepest layer
layer = 'conv5_block32_concat'; %Which layer to use??
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows');

%Extract labels from training/test
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%make template and fit IMAGE CLASSIFIER
% t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');

%Using defaults
% Default kernel is linear
% Should use linear if our data is LINEARLY seperateable
t = templateSVM('Standardize', true);

classifier = fitcecoc(featuresTrain, YTrain, 'Learners', t);

%% 
% 
%% Classify Test Images
% Classify the test images using the fine-tuned network, and calculate the test 
% accuracy.

%[YPred,probs] = classify(net,augimdsTest); I COMMENTED THIS OUT FOR THE
%SVM THING

% Extract the class labels from the test data.

YPred = predict(classifier, featuresTest);
YTest = imdsTest.Labels;

accuracy = mean(YPred == YTest);

% Calculate Average F-score.
s = max(size(unique(YTest)));
conf_values = confusionmat(YTest, YPred);
tp_m = diag(conf_values); 
for i = 1:s
    % True positive are the diagonals.
    TP = tp_m(i);
    % False positives are everything (excluding the TP) in the COLUMN.
    FP = sum(conf_values(:, i), 1) - TP;
    % False Negatives are everything (excluding the TP) in the ROW.
    FN = sum(conf_values(i, :), 2) - TP;
    % True Negatives are everything else not in the same ROW **OR** COLUMN
    TN = sum(conf_values(:)) - TP - FP - FN;
    % Calculate Recall
    Recall = TP ./(TP + FN);
    if isnan(Recall)
        Recall = 0;
    end
    % Calculate Precision
    Precision = TP ./ (TP + FP);
    if isnan(Precision)
        Precision = 0;
    end
    % Calculate FScore
    FScore = (2*(Precision * Recall)) / (Precision+Recall);
    if isnan(FScore)
        FScore = 0;
    end
 
    fscore(i) = FScore;
end
avg_f_score = mean(fscore(:));

% Display confusion chart.
fig = figure;
conf = confusionchart(YTest, YPred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position;
accuracy
fscore
avg_f_score
%% 
% Display four sample validation images with predicted labels and the predicted 
% probabilities of the images having those labels.

%idx = randperm(numel(imdsValidation.Files),4);
%figure
%for i = 1:4
    %subplot(2,2,i)
    %I = readimage(imdsValidation,idx(i));
    %imshow(I)
    %label = YPred(idx(i));
    %title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
%end

end

