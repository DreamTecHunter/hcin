clear;
wave_file = dir("wav_rec\*.wav");
for idx = 1:length(wave_file)
    fname = wave_file(idx).name;
    str = strfind(fname, '_');
    speaker_id = fname(1:str(1)-1);
    wave_file(idx).speaker_id = speaker_id;
    word_id = fname(str(1)+1:str(2)-1); 
    wave_file(idx).word_id = word_id;
    word_rep = fname(str(2)+1:end-4);
    wave_file(idx).word_rep = word_rep;
end
save("main.mat", "wave_file",'-mat')

%
% script_divide2subFol_speakers.m
%
% (c) 2022-11-30, co.falch-at-htlinn
%

clear
close all

in_fol = 'wav_rec\';
in_file = dir([in_fol,'*wav']);
n_file = length(in_file);

out_fol0 = 'wav_rec_divBySpkr\';
if ~isdir(out_fol0)
    mkdir(out_fol0)
end

out_fol1 = 'nn';
for i_file = 1:n_file
    nam = in_file(i_file).name;
    str = strfind(nam,'_');
    spkr = nam(1:str(1)-1);
    if strcmp([spkr,'\'],out_fol1)
    else
        out_fol1 = [spkr,'\'];
        mkdir([out_fol0,out_fol1])
    end
    copyfile([in_fol,nam],[out_fol0,out_fol1,nam])
end






<
% eof

clear
close all
clc

%database object creation
disp('(1) Database Object Creation')
a = audioDatastore('wav_rec', 'includesubfolders', true);
[aTrain, aTest] = splitEachLabel(a, 0.8);
trainDatastoreCount = countEachLabel(aTrain);
testDatastoreCount = countEachLabel(aTest);


reset(aTrain)

% feature extraction
disp('(2) feature extraction')
l_win = 0.03;
l_ol = 0.0025;
pov_thr = -40;
rcr_thr = 1000;

features = [];
labels = [];
energyThreshold = 0.005;
zcrThreshold = 0.2;

keepLen = round(length(aTrain));

while hasdata(aTrain)
    [audioIn, dsInfo] = read(adsTrain);

    audioIn = audioIn(1:keepLen);
    feat = extract(aTest, audioIn);
    isSpeech = feat(:, featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = feat(:, featureMap.zerocrossrate) < zcrThreshold;
    voicedSpeech = isSpeech & isVoiced;
    feat(~voicedSpeech, :) = [];
    feat(:, [featureMap.zerocrossrate, featureMap.shorTimeEnergy]) = [];
    label = repelem(dsInfro.Label, size(feat, 1));
    features = [features;feat];
    labels = [labels, label];
end


