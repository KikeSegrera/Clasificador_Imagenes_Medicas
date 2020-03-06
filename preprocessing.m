clc; clear all;	close all;

%% Carga de imagen
orgrgb = imread("I001.tif");
% [ren col bands] = size(orgrgb);
% set(gcf, 'Position', get(0, 'ScreenSize')); 
% set(gcf,'name','Análisis de grasa','numbertitle','off') 
% subplot(1, 2, 1); imshow(orgrgb);
% figure; imshow(orgrgb);
% figure; imshow(orgrgb(:,:,1));
% figure; imshow(orgrgb(:,:,2));
% figure; imshow(orgrgb(:,:,3));

orghsv = rgb2hsv(orgrgb);
channelH = orghsv(:,:,1);
channelH = histeq(channelH);
channelV = orghsv(:,:,3);
channelV = histeq(channelV);
channelS = orghsv(:,:,2);
channelS = histeq(channelS);
% figure; imshow(orghsv);
% figure; imshow(channelH);
% figure; imshow(channelV);
% figure; imshow(channelS);


%% Fibrosis
selectionMask = createMask(orgrgb,3);
% hueMask = channelH <= (mean(channelH(selectionMask)) + std(channelH(selectionMask)));
saturationMask = channelS <= (mean(channelS(selectionMask)) + 2*std(channelS(selectionMask)));
valueMask = channelV <= (mean(channelV(selectionMask)) + std(channelV(selectionMask)));

finalMask = saturationMask & valueMask;
finalMask = bwareaopen(finalMask,100);
se = strel('disk',2);
finalMask = imclose(finalMask,se);


segmented = orgrgb .* uint8(finalMask);
% subplot(1, 2, 2); imshow(segmented);
figure; imshow(segmented);
imwrite(segmented,"Class1.tif");

%% Estaeatosis
selectionMask = createMask(orgrgb,3);
% hueMask = channelH <= (mean(channelH(selectionMask)) + std(channelH(selectionMask)));
% saturationMask = channelS <= (mean(channelS(selectionMask)) + std(channelS(selectionMask)));
valueMask = channelV >= (mean(channelV(selectionMask)) - 3 * std(channelV(selectionMask)));

menor = 100;
mayor = 5000;
finalMaskBlobs = xor(bwareaopen(valueMask,menor), bwareaopen(valueMask,mayor));
se = strel('disk',1);
finalMaskBlobs = imclose(finalMaskBlobs,se);

% [labeledImage numberOfBlobs] = bwlabel(finalMaskBlobs, 8);
blobMeasurements = regionprops(finalMaskBlobs,'Centroid','Eccentricity','PixelIdxList');
[numberOfBlobs, ~] = size(blobMeasurements);
for i = 1:numberOfBlobs
    if blobMeasurements(i).Eccentricity > 0.70
        finalMaskBlobs(blobMeasurements(i).PixelIdxList) = 0;
    end
end
segmentedBlobs = orgrgb .* uint8(finalMaskBlobs);
figure; imshow(segmentedBlobs);
imwrite(segmentedBlobs,"Class2.tif");

completeMask = finalMask | finalMaskBlobs;
background = orgrgb .* uint8(~completeMask);
figure; imshow(background);
imwrite(background,"Class3.tif");

%% Funciones
function [mask] = createMask(rgbImage,areas)
    try

        mask = zeros(size(rgbImage(:,:,1)));
        %selections = cell(1,areas);

        hTemp = figure;
        imshow(rgbImage);
        set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.

        for h=1:areas
            message = sprintf('Selecciona el área de interés: %d ',h);
            uiwait(msgbox(message));
            hFH = drawfreehand(gca);
            % Se crea la máscara de lo que el usuario seleccionó
            mask = mask + hFH.createMask();
            %selections{h} = mask;
        end
        mask = logical(mask);
        % Se cierra la imágen abierta
        close(hTemp);

    catch ME
        errorMessage = sprintf('Error al seleccionar región', ...
            ME.message);
        WarnUser(errorMessage);
    end
end