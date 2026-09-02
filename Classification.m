% Define the path to the training and testing image folder and JSON file
trainFolder = 'dataset_classification/train'; 
trainJsonFile = 'dataset_classification/train/_groundtruth.json';
testFolder = 'dataset_classification/test'; 
testJsonFile = 'dataset_classification/test/_groundtruth.json';

% Load Training Data
trainJsonData = jsondecode(fileread(trainJsonFile));
numTrainSamples = numel(trainJsonData);

% Initialize training data
XTrain = [];
yTrain = cell(numTrainSamples, 1);  % Use cell array to avoid type mismatch

for idx = 1:numTrainSamples
    filename = trainJsonData(idx).filename;
    label = trainJsonData(idx).label;
    
    imgPath = fullfile(trainFolder, filename);
    if ~isfile(imgPath)
        error('Image file not found: %s', imgPath);
    end
    
    img = imread(imgPath);
    img = rgb2gray(img);
    img = imresize(img, [64 64]);

    % Extract features (HOG features)
    features = extractHOGFeatures(img);
    
    XTrain = [XTrain; features];  % Append feature vector
    yTrain{idx} = label;          % Store label as a cell element
end

% Convert labels to categorical
yTrain = categorical(yTrain);

% Load Test Data
testJsonData = jsondecode(fileread(testJsonFile));
numTestSamples = numel(testJsonData);

% Initialize test data
XTest = [];
yTest = cell(numTestSamples, 1);

for idx = 1:numTestSamples
    filename = testJsonData(idx).filename;
    label = testJsonData(idx).label;
    
    imgPath = fullfile(testFolder, filename);
    if ~isfile(imgPath)
        error('Image file not found: %s', imgPath);
    end
    
    img = imread(imgPath);
    img = rgb2gray(img);
    img = imresize(img, [64 64]);
    
    % Extract features
    features = extractHOGFeatures(img);
    
    XTest = [XTest; features];
    yTest{idx} = label;
end

% Convert labels to categorical
yTest = categorical(yTest);

% 1. Support Vector Machine (SVM) Classification
t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
svmModel = fitcecoc(XTrain, yTrain, 'Learners', t);

% 2. Random Forest Classification
rfModel = TreeBagger(200, XTrain, yTrain, 'Method', 'classification');

% 3. Neural Network Classification
XTrainNN = XTrain';  

% Convert labels to one-hot encoding
numClasses = numel(unique(yTrain));
targetTrain = full(ind2vec(double(yTrain)' + 1));  

hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
net.trainParam.showWindow = false;
net = train(net, XTrainNN, targetTrain);

% Evaluate models on the test set
svmPredictions = predict(svmModel, XTest);
svmAccuracy = mean(svmPredictions == yTest) * 100;
fprintf('SVM Accuracy: %.2f%%\n', svmAccuracy);

rfPrediction = predict(rfModel, XTest);
rfPrediction = categorical(rfPrediction);
rfAccuracy = mean(rfPrediction == yTest) * 100;
fprintf('Random Forest Accuracy: %.2f%%\n', rfAccuracy);

XTestNN  = XTest';
nnOutputs = net(XTestNN);
[~, nnPredictions] = max(nnOutputs, [], 1);
nnPredictions = nnPredictions - 1;
nnAccuracy = mean(nnPredictions' == double(yTest)) * 100;
fprintf('Neural Network Accuracy: %.2f%%\n', nnAccuracy);


% Select an Image for Prediction Using the Trained Models
choice = input('Do you want to select an image for prediction? (y/n): ', 's');
if lower(choice) == 'y'
    [file, path] = uigetfile('dataset_classification/*.jpg', 'Select an image');
    if isequal(file, 0)
         disp('No image selected.');
    else
         imagePath = fullfile(path, file);
         testImg = imread(imagePath);
         if size(testImg, 3) == 3
             testImg = rgb2gray(testImg);
         end
         testImg = imresize(testImg, [64 64]);
         
         % Extract features from the selected image.
         testFeatures = extractHOGFeatures(testImg);
         
         % SVM Prediction
         svmPrediction = predict(svmModel, testFeatures);
         
         % Random Forest Prediction
         rfPrediction = predict(rfModel, testFeatures);
         rfPrediction = categorical(rfPrediction);
         
         % Neural Network Prediction (the NN expects a column vector)
         nnInput = testFeatures';
         nnOutput = net(nnInput);
         [~, nnPredIdx] = max(nnOutput, [], 1);
         nnPrediction = nnPredIdx - 1;
         
         % Attempt to retrieve the ground truth label from the JSON files.
         truthLabel = '';
         % First search in the test JSON data
         for i = 1:numTestSamples
             if strcmp(testJsonData(i).filename, file)
                 truthLabel = testJsonData(i).label;
                 break;
             end
         end
         % If not found, search in the training JSON data
         if isempty(truthLabel)
             for i = 1:length(trainJsonData)
                 if strcmp(trainJsonData(i).filename, file)
                     truthLabel = trainJsonData(i).label;
                     break;
                 end
             end
         end
         if isempty(truthLabel)
             truthLabel = 'Not Available';
         end
         
         % Display the predictions and the ground truth label.
         fprintf('\nPredictions for the selected image:\n');
         fprintf('Ground Truth Label: %s\n', truthLabel);
         fprintf('SVM Prediction: %s\n', char(svmPrediction));
         fprintf('Random Forest Prediction: %s\n', char(rfPrediction));
         
         % Neural Network Prediction
         if nnPrediction == 1
             fprintf('Neural Network Prediction: Glioma\n');
         elseif nnPrediction == 2
             fprintf('Neural Network Prediction: Meningioma\n'); 
         elseif nnPrediction == 3
             fprintf('Neural Network Prediction: No Tumor\n'); 
         elseif nnPrediction == 4
             fprintf('Neural Network Prediction: Pituitary\n'); 
         else
             fprintf('Neural Network Prediction: %d\n', nnPrediction);
         end
    end
end
