%Pitch

%part 01

[audioIn, fs] = audioread("wav_rec_divBySpkr\\tobi\\tobi_ja_1.wav");
start = 110e3;
stop = 135e3;
s = start:stop;
timeVector = linspace(start/fs, stop/fs, numel(audioIn));

%sound(audioIn,fs)

figure
plot(timeVector,audioIn)
axis([(start/fs) (stop/fs) -1 1])
ylabel("Amplitude")
xlabel("Time (s)")
title("Utterance - Two")

%part 02

windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);

f0 = pitch(audioIn,fs,WindowLength=windowLength,OverlapLength=overlapLength,Range=[50,250]);

figure
subplot(2,1,1)
plot(timeVector,audioIn)
axis([(110e3/fs) (135e3/fs) -1 1])
ylabel("Amplitude")
xlabel("Time (s)")
title("Utterance - Two")

subplot(2,1,2)
timeVectorPitch = linspace(start/fs,stop/fs,numel(f0));
plot(timeVectorPitch,f0,"*")
axis([(110e3/fs) (135e3/fs) min(f0) max(f0)])
ylabel("Pitch (Hz)")
xlabel("Time (s)")
title("Pitch Contour")

%part 03

energyThreshold = 20;
[segments,~] = buffer(audioIn,windowLength,overlapLength,"nodelay");
ste = sum((segments.*hamming(windowLength,"periodic")).^2,1);
isSpeech = ste(:) > energyThreshold;

%part 04

zcrThreshold = 0.02;
zcr = zerocrossrate(audioIn,WindowLength=windowLength,OverlapLength=overlapLength);
isVoiced = zcr < zcrThreshold;

%part 05

voicedSpeech = isSpeech & isVoiced;

%part 06 

%f0(~voicedSpeech) = NaN;

figure
subplot(2,1,1)
plot(timeVector,audioIn)
axis([(110e3/fs) (135e3/fs) -1 1])
axis tight
ylabel("Amplitude")
xlabel("Time (s)")
title("Utterance - Two")

subplot(2,1,2)
plot(timeVectorPitch,f0,"*")
axis([(110e3/fs) (135e3/fs) min(f0) max(f0)])
ylabel("Pitch (Hz)")
xlabel("Time (s)")
title("Pitch Contour")

%data set

%part 01

ads = audioDatastore('wav_rec_divBySpkr','IncludeSubfolders',true, 'LabelSource','foldernames');

%part 02

[adsTrain,adsTest] = splitEachLabel(ads,0.8);

%part 03

adsTrain;

%part 04

trainDatastoreCount = countEachLabel(adsTrain);

%part 05

[sampleTrain,dsInfo] = read(adsTrain);
sound(sampleTrain,dsInfo.SampleRate)

%part 06

reset(adsTrain);

%Feature Extraction

%part 01

fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);
afe = audioFeatureExtractor(SampleRate=fs, ...
    Window=hamming(windowLength,"periodic"),OverlapLength=overlapLength, ...
    zerocrossrate=true,shortTimeEnergy=true,pitch=true,mfcc=true);

%part 02 

featureMap = info(afe)

%part 03

features = [];
labels = [];
energyThreshold = 0.005;
zcrThreshold = 0.2;

keepLen = round(length(sampleTrain)/3);

while hasdata(adsTrain)
    [audioIn,dsInfo] = read(adsTrain);

    % Take the first portion of each recording to speed up code
    audioIn = audioIn(1:keepLen);

    feat = extract(afe,audioIn);
    isSpeech = feat(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = feat(:,featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    feat(~voicedSpeech,:) = [];
    feat(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    label = repelem(dsInfo.Label,size(feat,1));
    
    features = [features;feat];
    labels = [labels,label];
end

%part 04

M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;

% Training a CLassifier

%part 01

trainedClassifier = fitcknn(features,labels, ...
    Distance="euclidean", ...
    NumNeighbors=5, ...
    DistanceWeight="squaredinverse", ...
    Standardize=false, ...
    ClassNames=unique(labels));

%part 02

k = 5;
group = labels;
c = cvpartition(group,KFold=k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,CVPartition=c);

%part 03

validationAccuracy = 1 - kfoldLoss(partitionedModel,LossFun="ClassifError");
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

%part 04 

validationPredictions = kfoldPredict(partitionedModel);
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels,validationPredictions,title="Validation Accuracy", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

% Testing the Classifier

%part 01

features = [];
labels = [];
numVectorsPerFile = [];
while hasdata(adsTest)
    [audioIn,dsInfo] = read(adsTest);
    
    % Take the same first portion of each recording to speed up code
    audioIn = audioIn(1:keepLen);

    feat = extract(afe,audioIn);

    isSpeech = feat(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = feat(:,featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    feat(~voicedSpeech,:) = [];
    numVec = size(feat,1);
    feat(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    
    label = repelem(dsInfo.Label,numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features = [features;feat];
    labels = [labels,label];
end
features = (features-M)./S;


%part 02

prediction = predict(trainedClassifier,features);
prediction = categorical(string(prediction));

%part 03 

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels(:),prediction,title="Test Accuracy (Per Frame)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

%part 04 

r2 = prediction(1:numel(adsTest.Files));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(adsTest.Labels,r2,title="Test Accuracy (Per File)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

