clc; clear all;	close all;

%% Feature extraction
nSamples = 1000;
F = [];
L = [];
sigmas = [1 2 3];

images = {};
images{1} = imread("I001_Class1.tif");
images{2} = imread("I001_Class2.tif");
images{3} = imread("I001_Class3.tif");
images{4} = imread("I002_Class1.tif");
images{5} = imread("I002_Class2.tif");
images{6} = imread("I002_Class3.tif");

classes = {};
classes{1} = 1;
classes{2} = 2;
classes{3} = 3;
classes{4} = 1;
classes{5} = 2;
classes{6} = 3;

for i = 1:length(images)
    [Faux Laux] = featureExtraction(images{i},classes{i},nSamples,sigmas);
    F = [F; Faux];
    L = [L; Laux];
end

%% Training
% Random forest
[treeBag, featImp, oobPredError] = rfTrain(F,L,65,60);
save('RF','treeBag');

%Naive Bayes
bayes = fitcnb(F,L);
save('NB','bayes');

%LDA
lda = fitcdiscr(F,L);
save('LDA','lda');
%% Funciones
function [features, labels] = featureExtraction(rgbImage, class, nSamples, sigmas)
    labels = class .* ones(nSamples,1);
    %Features
    F = [];
    hsvImage = rgb2hsv(rgbImage);
    bwImage = rgb2gray(rgbImage);
    R = rgbImage(:,:,1);
    G = rgbImage(:,:,2);
    B = rgbImage(:,:,3);
    H = hsvImage(:,:,1);
    S = hsvImage(:,:,2);
    V = hsvImage(:,:,3);
    
    for sigma = sigmas
        D = zeros(size(S,1),size(S,2),8);
        [D(:,:,1),D(:,:,2),D(:,:,3),D(:,:,4),D(:,:,5),D(:,:,6),D(:,:,7),D(:,:,8)] = derivatives(V,sigma);
        F = cat(3,F,D);
        F = cat(3,F,sqrt(D(:,:,2).^2+D(:,:,3).^2)); % edges
    end
    
    %Samples
    random = zeros(size(bwImage));
    random = imnoise(random,'salt & pepper',0.55);
    
    samples = random & bwImage;
    indexes = find(samples == 1);
    
    sampleIndexes = indexes(randperm(length(indexes)));
    sampleIndexes = sampleIndexes(1:nSamples);
    
    features = [double(R(sampleIndexes)) double(G(sampleIndexes)) double(B(sampleIndexes)) H(sampleIndexes) S(sampleIndexes) V(sampleIndexes)];
    
    for i = 1:size(F,3)
        aux = F(:,:,i);
        features = [features aux(sampleIndexes)];
    end
end

function [treeBag,featImp,oobPredError] = rfTrain(rfFeat,rfLbl,ntrees,minleafsize)

opt = statset('UseParallel',true);
treeBag = TreeBagger(ntrees,rfFeat,rfLbl,'MinLeafSize',minleafsize,'oobvarimp','on','opt',opt);
if nargout > 1
    featImp = treeBag.OOBPermutedVarDeltaError;
end
if nargout > 2
    oobPredError = oobError(treeBag);
end

end
