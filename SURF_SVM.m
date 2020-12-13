
function SURF_SVM()

imdsTest = imageDatastore(fullfile('COVID19_CT_seg_20cases_DUMMY', 'test'),'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain = imageDatastore(fullfile('UCSD_combined', 'train'),'IncludeSubfolders',true,'LabelSource','foldernames');

bag = bagOfFeatures(imdsTrain);

imageForBarChart = readimage(imdsTrain,1);

featuresVector = encode(bag, imageForBarChart);
figure
bar(featuresVector)
title('freq histogram training patient 1')
xlabel('visual word index')
ylabel('word count')


optionsSVM = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true)


categoryClassifier = trainImageCategoryClassifier(imdsTrain, bag,'LearnerOptions', optionsSVM);


disp("EVALUATING WITH CLASSIFIER AND IMDSTRAIN FOR SANITY")
confMatrix_train = evaluate(categoryClassifier, imdsTrain);
train_ACC = mean(diag(confMatrix_train))

disp("EVALUATING WITH CLASSIFIER AND TEST FOR PROOF")
confMatrix_test = evaluate(categoryClassifier, imdsTest);
test_ACC = mean(diag(confMatrix_test))

disp("<------------------------------------------------------------------------------>")

disp("USING training IMAGES FOR TESTING (SHOULD BE A HIGH ACCURACY)")

disp("Testing test image 1 on classifier")
Test = rgb2gray(readimage(imdsTrain,1));
Test2 = rgb2gray(readimage(imdsTrain,2));
Test3 = rgb2gray(readimage(imdsTrain,3));
%     currentIMG_gray = single(rgb2gray(currentIMG));


disp("USING THE TESTING IMAGES FOR TESTING")
[labelIdx, scores] = predict(categoryClassifier, Test);
% Display the string label
categoryClassifier.Labels(labelIdx)
disp("CORRECT LABEL BELOW:")
imdsTrain.Labels(1)







disp("Testing test image 2 on classifier")
[labelIdx, scores] = predict(categoryClassifier, Test2);
% Display the string label
categoryClassifier.Labels(labelIdx)
disp("CORRECT LABEL BELOW:")

imdsTrain.Labels(2)

disp("Testing test image 3 on classifier")
[labelIdx, scores] = predict(categoryClassifier, Test3);
% Display the string label
categoryClassifier.Labels(labelIdx)
disp("CORRECT LABEL BELOW:")

imdsTrain.Labels(3)
disp("<------------------------------------------------------------------------------>")
disp("USING testing IMAGES FOR TESTING")
disp("USING testing IMAGES FOR predict (after extracting features from training")




disp("Testing test image 1 on classifier")
Test = rgb2gray(readimage(imdsTest,1));
Test2 = rgb2gray(readimage(imdsTest,2));
Test3 = rgb2gray(readimage(imdsTest,3));


disp("USING THE TESTING IMAGES FOR TESTING")
[labelIdx, scores] = predict(categoryClassifier, Test);
% Display the string label
categoryClassifier.Labels(labelIdx)
disp("CORRECT LABEL BELOW:")
imdsTest.Labels(1)


disp("Testing test image 2 on classifier")
[labelIdx, scores] = predict(categoryClassifier, Test2);
% Display the string label
categoryClassifier.Labels(labelIdx)
disp("CORRECT LABEL BELOW:")

imdsTest.Labels(2)




disp("Testing test image 3 on classifier")
[labelIdx, scores] = predict(categoryClassifier, Test3);
% Display the string label
categoryClassifier.Labels(labelIdx)
disp("CORRECT LABEL BELOW:")

imdsTest.Labels(3)

disp("<------------------------------------------------------------------------------>")


SURF_TEST = rgb2gray(readimage(imdsTrain,1));


points = detectSURFFeatures(SURF_TEST);
imshow(SURF_TEST); hold on;
plot(points.selectStrongest(10));
title("Training image 1")


figure
imshow(SURF_TEST); hold on;
plot(points.selectStrongest(40));
title("Training image COVID")
imdsTrain.Labels(1)



disp("PREDICING FOR ALL OF IMDS_TEST WITH CLASSIFIER")
YPred = predict(categoryClassifier, imdsTest);
YTest = imdsTest.Labels;


disp("SIZE AND TYPE FOR YPRED AND YTEST")
disp("YPRED:")
size(YPred)
class(YPred)
disp("YTEST:")
size(YTest)
class(YTest)


% Calculate accuracy.
% accuracy = mean(YPred == YTest);

% Get confusion matrix values.
conf_values = confusionmat(YTest, YPred);
YPred_cat = categorical(YPred)
YPred_cat(YPred_cat == "1") = "covid";
YPred_cat(YPred_cat == "2") = "noncovid";

% Calculate accuracy.
accuracy = mean(YPred_cat == YTest);

% Get confusion matrix values.
conf_values = confusionmat(YTest, YPred_cat);

% Get true positive, false negative, false positive
TP = conf_values(1,1);
FN = conf_values(1,2);
FP = 0;
TN = 0;

combined_mat = [TP, FN ; FP, TN]
% Calculate Recall, precision, fscore
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

% Create confusion chart.
figure
conf = confusionchart(combined_mat,["Covid", "Non-covid"],'RowSummary','row-normalized','ColumnSummary','column-normalized');
% conf_values = confusionchart(YTest, YPred_cat,'RowSummary','row-normalized','ColumnSummary','column-normalized')

accuracy
FScore

end