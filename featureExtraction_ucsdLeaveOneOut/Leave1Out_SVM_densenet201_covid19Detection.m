function Leave1Out_SVM_densenet201_covid19Detection()

%% *Covid Detection Feature Extraction withcd leave one patient out validation*
%% Load Pretrained Network

net = densenet201;
%% 
% Use |analyzeNetwork| to display an interactive visualization of the network 
% architecture and detailed information about the network layers.
% analyzeNetwork(net);
%% 
% 

net.Layers(1);
inputSize = net.Layers(1).InputSize;
%% Performance Data Initialization and other variables

numPatients = 384;

% Matrix to hold performance data for each train and test run 
% columns (accuracy,fscore,avg_fscore)
performance = zeros([0,1]);
%% Load Data and Train Network for every One Patient Out Validation (384 patients total)

% Initialize true positive, true negative, false positive, false negative.
sum_TP = 0;
sum_TN = 0;
sum_FP = 0;
sum_FN = 0;

% Loop for doing each one patient out validation and then training the
% model and getting performance results
for patientNum = 1:numPatients


    [fullPatientList,isCOVID] = leaveOneOutSetup(patientNum); % Prep file structure and data ('leave_out') folder
    
    
    % Data splits: one patient in validation, the rest in training.
    imdsTrain = imageDatastore(fullfile('leave_out','train'),'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsValidation = imageDatastore(fullfile('leave_out','validation'),'IncludeSubfolders',true,'LabelSource','foldernames');
    
    %***Image augmentation with variability***%
    % Use an augmented image datastore to automatically resize the training images. 
    % Specify additional augmentation operations to perform on the training images: 
    % randomly flip the training images along the vertical axis and randomly translate them 
    % up to 30 pixels and scale them up to 10% horizontally and vertically. 
    % Data augmentation helps prevent the network from overfitting and memorizing 
    % the exact details of the training images.  
    
    %{
    pixelRange = [-30 30];
    scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange, ...
        'RandXScale',scaleRange, ...
        'RandYScale',scaleRange);
    
    augimdsTrain = augmentedImageDatastore(was thiinputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
    %}
    
    %  Standard image augmentation
    augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,"ColorPreprocessing","gray2rgb");
    augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation,"ColorPreprocessing","gray2rgb");
 
        
    

%% SVM Classifier, Feature Extraction

    %deepest layer
    layer = 'conv5_block32_concat'; %Which layer to use??
    featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
    featuresValidation = activations(net, augimdsValidation, layer, 'OutputAs', 'rows');
    
    %Extract labels from training/test
    YTrain = imdsTrain.Labels;
    YValidation = imdsValidation.Labels;
    
    t = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto', 'Standardize', true);
    
    classifier = fitcecoc(featuresTrain, YTrain, 'Learners', t);
    
    YPred = predict(classifier, featuresValidation);
%% Classify Test Images
% Classify the test images using the fine-tuned network, and calculate the test 
% accuracy.

%     [YPred,probs] = classify(net,augimdsValidation);

    % Extract the class labels from the test data.
    YTest = imdsValidation.Labels;
    
    % Calculate accuracy
    accuracy = mean(YPred == YTest);
    
    % Get confusion matrix values.
    conf_values = confusionmat(YTest, YPred);
    
    
    
    % Add to total categories (for COVID).
    if(isCOVID == 1)
        sum_TP = sum_TP + conf_values(1,1);
        TP = conf_values(1,1);
        sum_FN = sum_FN + conf_values(1,2);
        FN = conf_values(1,2);
        sum_FP = sum_FP + conf_values(2,1);
        FP = conf_values(2,1);
        sum_TN = sum_TN + conf_values(2,2);
        TN = conf_values(2,2);
        
        combined_mat = [TP, FN ; FP, TN];
%         figure
%         conf1 = confusionchart(combined_mat,["Covid", "Non-covid"],'RowSummary','row-normalized','ColumnSummary','column-normalized');
    end

    % Add to total categories (for nonCOVID) The confusion matrix for these patients
    % are reversed.
    if(isCOVID == 0)
        sum_TN = sum_TN + conf_values(1,1);
        TN = conf_values(1,1);
        sum_FP = sum_FP + conf_values(1,2);
        FP = conf_values(1,2);
        sum_FN = sum_FN + conf_values(2,1);
        FN = conf_values(2,1);
        sum_TP = sum_TP + conf_values(2,2);
        TP = conf_values(2,2);
        
        combined_mat = [TP, FN ; FP, TN];
%         figure
%         conf2 = confusionchart(combined_mat,["Covid", "Non-covid"],'RowSummary','row-normalized','ColumnSummary','column-normalized');
    end
 
    performance(patientNum, 1) = accuracy;
    
    % Print patient info and results
    fprintf('%d >> %s, ACC=%.3f\n',patientNum,fullPatientList{patientNum},accuracy)
end % End function for leave on out validation training on all patients
%% Ending performance stats

% Print ending performance matrix
disp('**************************************************************************')
disp('*                          Final Performance Report                      *')
disp('**************************************************************************')

% Combined CM
combined_mat = [sum_TP,  sum_FN; sum_FP, sum_TN]

% Display combined confusion chart.
figure;
conf = confusionchart(combined_mat,["Covid", "Non-covid"],'RowSummary','row-normalized','ColumnSummary','column-normalized');

% Avg Accuracy from total list of ACCs
P = mean(performance);
avgAccuracy = P(1)
sDev_accuracy = std(performance(:))

% Avg Accuracy from combined CM
combined_accuracy = (sum(diag(combined_mat))/sum(sum(combined_mat)))*100

% Calculate F1 Score
TP = combined_mat(1,1);
FN = combined_mat(1,2);
FP = combined_mat(2,1);
TN = combined_mat(2,2);
% Calculate Recall
Recall = TP /(TP + FN);
if isnan(Recall)
    Recall = 0;
end
% Calculate Precision
Precision = TP / (TP + FP);
if isnan(Precision)
    Precision = 0;
end
% Calculate FScore
FScore = (2*(Precision * Recall)) / (Precision+Recall);
if isnan(FScore)
    FScore = 0;
end

FScore

% save('leave_one_densenet201.mat','avgAccuracy','avgF1Score','sDev_accuracy','sDev_F1');

end