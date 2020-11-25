function LEAVEONEOUT_SVM_DENSE()
%% *Covid Detection Transfer Learning with leave one patient out validation (resnet50)*
%% Load Pretrained Network
% 

net = densenet201;
%% 
% Use |analyzeNetwork| to display an interactive visualization of the network 
% architecture and detailed information about the network layers.

% analyzeNetwork(net);
%% 
% 
%% 
% 

net.Layers(1);
inputSize = net.Layers(1).InputSize;
%% Replace Final Layers
% Extract the layer graph from the trained network. If the network is a |SeriesNetwork| 
% object, such as AlexNet, VGG-16, or VGG-19, then convert the list of layers 
% in |net.Layers| to a layer graph.
% 
% if isa(net,'SeriesNetwork') 
%   lgraph = layerGraph(net.Layers); 
% else
%   lgraph = layerGraph(net);
% end 
% %% 
% % Find the names of the two layers to replace. You can do this manually or you 
% % can use the supporting function <matlab:edit(fullfile(matlabroot,'examples','nnet','main','findLayersToReplace.m')) 
% % findLayersToReplace> to find these layers automatically. 
% 
% [learnableLayer,classLayer] = findLayersToReplace(lgraph);
% [learnableLayer,classLayer];
% %% 
% % In most networks, the last layer with learnable weights is a fully connected 
% % layer. Replace this fully connected layer with a new fully connected layer with 
% % the number of outputs equal to the number of classes in the new data set (5, 
% % in this example). In some networks, such as SqueezeNet, the last learnable layer 
% % is a 1-by-1 convolutional layer instead. In this case, replace the convolutional 
% % layer with a new convolutional layer with the number of filters equal to the 
% % number of classes. To learn faster in the new layer than in the transferred 
% % layers, increase the learning rate factors of the layer.
% 
% % numClasses = numel(categories(imdsTrain.Labels));
% numClasses = 2;
% 
% if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
%     newLearnableLayer = fullyConnectedLayer(numClasses, ...
%         'Name','new_fc', ...
%         'WeightLearnRateFactor',10, ...
%         'BiasLearnRateFactor',10);
%     
% elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
%     newLearnableLayer = convolution2dLayer(1,numClasses, ...
%         'Name','new_conv', ...
%         'WeightLearnRateFactor',10, ...
%         'BiasLearnRateFactor',10);
% end
% 
% lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
% %% 
% % The classification layer specifies the output classes of the network. Replace 
% % the classification layer with a new one without class labels. |trainNetwork| 
% % automatically sets the output classes of the layer at training time. 
% 
% newClassLayer = classificationLayer('Name','new_classoutput');
% lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
%% 
% To check that the new layers are connected correctly, plot the new layer graph 
% and zoom in on the last layers of the network.

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph);
% ylim([0,10]);
% %% Freeze Initial Layers and Training options
% 
% layers = lgraph.Layers;
% connections = lgraph.Connections;
% 
% frozen = 1;
% layers(1:frozen) = freezeWeights(layers(1:frozen));
% lgraph = createLgraphUsingConnections(layers,connections);
% 
% % Training options
% miniBatchSize = 10;
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',5, ...
%     'InitialLearnRate',3e-4, ...
%     'Shuffle','every-epoch', ...
%     'Verbose',false);
%% Performance Data Initialization and other variables

numPatients = 384;

% Matrix to hold performance data for each train and test run 
% columns (accuracy,fscore,avg_fscore)
performance = zeros([numPatients,2]);

%% Load Data and Train Network for every One Patient Out Validation (384 patients total)

% Loop for doing each one patient out validation and then training the
% model and getting performance results
for patientNum = 1:numPatients


    fullPatientList = leaveOneOutSetup(patientNum); % Prep file structure and data ('leave_out') folder
    
    
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
    
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
    %}
    
    %  Standard image augmentation
    augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,"ColorPreprocessing","gray2rgb");
    augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation,"ColorPreprocessing","gray2rgb");
% augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,"ColorPreprocessing","gray2rgb");
% augimdsTest = augmentedImageDatastore(inputSize,imdsTest,"ColorPreprocessing","gray2rgb");


   
    %  Train network   
%     net = trainNetwork(augimdsTrain,lgraph,options);
    



% *SVM CLASSIFIER*
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
    
%     Tabulate the results using a confusion matrix.
%     confMat = confusionmat(YTest, YPred);
    
%     Display the mean accuracy
%     accuracy = (sum(diag(confMat))/sum(sum(confMat)))*100;
%     sensitivity = (confMat(2,2)/(confMat(2,1)+confMat(2,2)))*100; % COVID
%     specificity = (confMat(1,1)/(confMat(1,1)+confMat(1,2)))*100; % NON-COVID
%     f1score = (2*confMat(2,2))/(2*confMat(2,2)+confMat(2,1)+confMat(1,2));
%     
%     ACC=[ACC,accuracy];
%     SEN=[SEN,sensitivity];
%     SPE=[SPE,specificity];
%     F1 = [F1,f1score];
    
    
    % Display confusion chart and patient number
%     fig = figure;
%     conf = confusionchart(YTest, YPred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
%     fig_Position = fig.Position;
%     fig_Position(3) = fig_Position(3)*1.5;
%     fig.Position = fig_Position;
    
    performance(patientNum, 1) = accuracy;
    performance(patientNum, 2) = avg_f_score;
    
    % Print patient info and results
   fprintf('%d >> %s, ACC=%.3f, F1=%.3f\n',patientNum,fullPatientList{patientNum},accuracy,avg_f_score)

    
%%
end % End function for leave on out validation training on all patients
%% Ending performance stats

% Print ending performance matrix
disp('**************************************************************************')
disp('*                          Final Perfromance Report                      *')
disp('**************************************************************************')

performance

P = mean(performance);
avgAccuracy = P(1)
avgF1Score = P(2)


sDev_accuracy = std(performance(:,1))
sDev_F1 = std(performance(:,2))





%% 
%
end