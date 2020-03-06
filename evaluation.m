clc; clear all;	close all;

%% Carga de Imágenes
gt = {};
gt{1} = imread("I003_Class1.tif");
gt{2} = imread("I003_Class2.tif");
gt{3} = imread("I003_Class3.tif");

lda = {};
lda{1} = imread("I003_Class1_LDA.tif");
lda{2} = imread("I003_Class2_LDA.tif");
lda{3} = imread("I003_Class3_LDA.tif");

rf = {};
rf{1} = imread("I003_Class1_RF.tif");
rf{2} = imread("I003_Class2_RF.tif");
rf{3} = imread("I003_Class3_RF.tif");

nb = {};
nb{1} = imread("I003_Class1_NB.tif");
nb{2} = imread("I003_Class2_NB.tif");
nb{3} = imread("I003_Class3_NB.tif");

[r, c, ~] = size(gt{1});

%% Etiquetado de clases
for i = 1:length(gt)
    gt{i} = rgb2gray(gt{i});
    lda{i} = rgb2gray(lda{i});
    rf{i} = rgb2gray(rf{i});
    nb{i} = rgb2gray(nb{i});
    
    gt{i} = gt{i} > 0;
    lda{i} = lda{i} > 0;
    rf{i} = rf{i} > 0;
    nb{i} = nb{i} > 0;
    
    gt{i} = i .* gt{i};
    lda{i} = i .* lda{i};
    rf{i} = i .* rf{i};
    nb{i} = i .* nb{i};
end

%% Creación de imágenes etiquetadas
gt_labeled = gt{1} + gt{2} + gt{3};
lda_labeled = lda{1} + lda{2} + lda{3};
rf_labeled = rf{1} + rf{2} + rf{3};
nb_labeled = nb{1} + nb{2} + nb{3};

gt_vector = reshape(gt_labeled,[r*c, 1]);
lda_vector = reshape(lda_labeled,[r*c, 1]);
rf_vector = reshape(rf_labeled,[r*c, 1]);
nb_vector = reshape(nb_labeled,[r*c, 1]);

%% Matrices de confusión
CLDA = confusionmat(gt_vector,lda_vector);
CRF = confusionmat(gt_vector,rf_vector);
CNB = confusionmat(gt_vector,nb_vector);

figure; confusionchart(CLDA); title('LDA');
figure; confusionchart(CRF); title('Random Forest');
figure; confusionchart(CNB); title('Naive Bayes');

%% Estadísticos matriz de confusión
%Accuracy
AccLDA = (CLDA(1,1) + CLDA(2,2) + CLDA(3,3))/sum(CLDA(:));
AccRF = (CRF(1,1) + CRF(2,2) + CRF(3,3))/sum(CRF(:));
AccNB = (CNB(1,1) + CNB(2,2) + CNB(3,3))/sum(CNB(:));

%Precision
PLDA1 = CLDA(1,1)/(CLDA(1,1) + CLDA(2,1) + CLDA(3,1));
PLDA2 = CLDA(2,2)/(CLDA(1,2) + CLDA(2,2) + CLDA(3,2));
PLDA3 = CLDA(3,3)/(CLDA(1,3) + CLDA(2,3) + CLDA(3,3));

PRF1 = CRF(1,1)/(CRF(1,1) + CRF(2,1) + CRF(3,1));
PRF2 = CRF(2,2)/(CRF(1,2) + CRF(2,2) + CRF(3,2));
PRF3 = CRF(3,3)/(CRF(1,3) + CRF(2,3) + CRF(3,3));

PNB1 = CNB(1,1)/(CNB(1,1) + CNB(2,1) + CNB(3,1));
PNB2 = CNB(2,2)/(CNB(1,2) + CNB(2,2) + CNB(3,2));
PNB3 = CNB(3,3)/(CNB(1,3) + CNB(2,3) + CNB(3,3));

%Recall
RLDA1 = CLDA(1,1)/(CLDA(1,1) + CLDA(1,2) + CLDA(1,3));
RLDA2 = CLDA(2,2)/(CLDA(2,1) + CLDA(2,2) + CLDA(2,3));
RLDA3 = CLDA(3,3)/(CLDA(3,1) + CLDA(3,2) + CLDA(3,3));

RRF1 = CRF(1,1)/(CRF(1,1) + CRF(1,2) + CRF(1,3));
RRF2 = CRF(2,2)/(CRF(2,1) + CRF(2,2) + CRF(2,3));
RRF3 = CRF(3,3)/(CRF(3,1) + CRF(3,2) + CRF(3,3));

RNB1 = CNB(1,1)/(CNB(1,1) + CNB(1,2) + CNB(1,3));
RNB2 = CNB(2,2)/(CNB(2,1) + CNB(2,2) + CNB(2,3));
RNB3 = CNB(3,3)/(CNB(3,1) + CNB(3,2) + CNB(3,3));

%F1 score
FLDA1 = (2 * PLDA1 * RLDA1)/(PLDA1 + RLDA1);
FLDA2 = (2 * PLDA2 * RLDA2)/(PLDA2 + RLDA2);
FLDA3 = (2 * PLDA3 * RLDA3)/(PLDA3 + RLDA3);
FLDA = (FLDA1 + FLDA2 + FLDA3)/3;

FRF1 = (2 * PRF1 * RRF1)/(PRF1 + RRF1);
FRF2 = (2 * PRF2 * RRF2)/(PRF2 + RRF2);
FRF3 = (2 * PRF3 * RRF3)/(PRF3 + RRF3);
FRF = (FRF1 + FRF2 + FRF3)/3;

FNB1 = (2 * PNB1 * RNB1)/(PNB1 + RNB1);
FNB2 = (2 * PNB2 * RNB2)/(PNB2 + RNB2);
FNB3 = (2 * PNB3 * RNB3)/(PNB3 + RNB3);
FNB = (FNB1 + FNB2 + FNB3)/3;