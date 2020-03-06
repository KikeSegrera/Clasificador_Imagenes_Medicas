clc; clear all;	close all;

%% Feature extraction
F = [];
L = [];
sigmas = [1 2 3];

images = {};
images{1} = imread("I003.tif");
images{2} = imread("I004.tif");

for i = 1:length(images)
    [Faux] = featureExtractionTest(images{i},sigmas);
    F = [F; Faux];
end

%% Testing
%Random Forest
load('RF.mat');
[~, scores] = predict(treeBag,F);
[~,indOfMax] = max(scores,[],2);

imL1RF = reshape(indOfMax(1:1228800),[960,1280]);
imL2RF = reshape(indOfMax(1228801:2457600),[960,1280]);

imwrite(images{1} .* uint8(imL1RF == 1),"I003_Class1_RF.tif");
imwrite(images{1} .* uint8(imL1RF == 2),"I003_Class2_RF.tif");
imwrite(images{1} .* uint8(imL1RF == 3),"I003_Class3_RF.tif");

imwrite(images{2} .* uint8(imL2RF == 1),"I004_Class1_RF.tif");
imwrite(images{2} .* uint8(imL2RF == 2),"I004_Class2_RF.tif");
imwrite(images{2} .* uint8(imL2RF == 3),"I004_Class3_RF.tif");

%Naive Bayes
load('NB.mat');
[~, scores] = predict(bayes,F);
[~,indOfMax] = max(scores,[],2);

imL1NB = reshape(indOfMax(1:1228800),[960,1280]);
imL2NB = reshape(indOfMax(1228801:2457600),[960,1280]);

imwrite(images{1} .* uint8(imL1NB == 1),"I003_Class1_NB.tif");
imwrite(images{1} .* uint8(imL1NB == 2),"I003_Class2_NB.tif");
imwrite(images{1} .* uint8(imL1NB == 3),"I003_Class3_NB.tif");

imwrite(images{2} .* uint8(imL2NB == 1),"I004_Class1_NB.tif");
imwrite(images{2} .* uint8(imL2NB == 2),"I004_Class2_NB.tif");
imwrite(images{2} .* uint8(imL2NB == 3),"I004_Class3_NB.tif");

%LDA
load('LDA.mat');
[~, scores] = predict(lda,F);
[~,indOfMax] = max(scores,[],2);

imL1LDA = reshape(indOfMax(1:1228800),[960,1280]);
imL2LDA = reshape(indOfMax(1228801:2457600),[960,1280]);

imwrite(images{1} .* uint8(imL1LDA == 1),"I003_Class1_LDA.tif");
imwrite(images{1} .* uint8(imL1LDA == 2),"I003_Class2_LDA.tif");
imwrite(images{1} .* uint8(imL1LDA == 3),"I003_Class3_LDA.tif");

imwrite(images{2} .* uint8(imL2LDA == 1),"I004_Class1_LDA.tif");
imwrite(images{2} .* uint8(imL2LDA == 2),"I004_Class2_LDA.tif");
imwrite(images{2} .* uint8(imL2LDA == 3),"I004_Class3_LDA.tif");

%% Funciones
function [features] = featureExtractionTest(rgbImage, sigmas)
    [r, c, ~] = size(rgbImage);
    
    F = [];
    hsvImage = rgb2hsv(rgbImage);
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
    
    indexes = linspace(1,r*c,r*c);
    indexes = indexes';
    
    features = [double(R(indexes)) double(G(indexes)) double(B(indexes)) H(indexes) S(indexes) V(indexes)];
    
    for i = 1:size(F,3)
        aux = F(:,:,i);
        features = [features aux(indexes)];
    end
end
