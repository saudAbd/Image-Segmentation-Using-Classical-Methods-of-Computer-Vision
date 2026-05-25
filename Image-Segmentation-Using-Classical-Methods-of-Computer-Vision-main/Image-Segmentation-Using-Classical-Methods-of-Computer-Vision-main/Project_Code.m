clc; clear; close all;

% Define dataset folder path
dataFolder = 'ball_frames'; 

% output folder directory
outputFolder = 'output';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get list of all image files
allFiles = dir(fullfile(dataFolder, '*.png'));

% Separate RGB images and mask images
rgbFiles = allFiles(~contains({allFiles.name}, '_GT')); 
gtFiles = allFiles(contains({allFiles.name}, '_GT'));  

% Ensure equal number of RGB images and ground truth masks
if length(rgbFiles) ~= length(gtFiles)
    error('Mismatch %d Image', length(rgbFiles), length(gtFiles));
end

% Extract numeric parts for proper sorting
rgbNums = regexp({rgbFiles.name}, 'frame-(\d+).png', 'tokens');
gtNums = regexp({gtFiles.name}, 'frame-(\d+)_GT.png', 'tokens');

rgbNums = cellfun(@(x) str2double(x{1}), rgbNums);
gtNums = cellfun(@(x) str2double(x{1}), gtNums);

% Sort filenames numerically
[~, rgbSortIdx] = sort(rgbNums);
[~, gtSortIdx] = sort(gtNums);

rgbFiles = rgbFiles(rgbSortIdx);
gtFiles = gtFiles(gtSortIdx);

disp('program starts......');

% Initialize DSC scores array
dscScores = zeros(length(rgbFiles), 1);

% Store segmented images, GT masks, and frame names
segmentedImages = cell(length(rgbFiles),1);
gtMasks = cell(length(gtFiles),1);
frameNames = cell(length(rgbFiles),1);

for i = 1:length(rgbFiles)
    % Read RGB image
    rgbImage = imread(fullfile(dataFolder, rgbFiles(i).name));
    gtMask = imread(fullfile(dataFolder, gtFiles(i).name));

    % Convert mask to binary
    if size(gtMask, 3) > 1
        gtMask = rgb2gray(gtMask);
    end
    gtMask = imbinarize(gtMask);
   
    % step 1. Convert to Grayscale
    grayImage = rgb2gray(rgbImage); 

    % step 2. Convert to Gaussian filter
    filtered_image = imgaussfilt(grayImage, 1);

    % step 4: Apply Adaptive Thresholding
    level = adaptthresh(filtered_image, 0.16);

    % step 5: Binarize the Image
    binaryImage = imbinarize(filtered_image, level);   

    % step 6: Remove detected objects above the floor level (wall removal)
    [height, width] = size(binaryImage);
    floorBoundary = round(height * 0.55 - (0.15 * (1:width) / width * height));
    for x = 1:width
        binaryImage(1:floorBoundary(x), x) = 0; 
    end

    % step 7: HSV-based segmentation for the orange ball
    hsvImage = rgb2hsv(rgbImage);
    lowerOrange = [0.02, 0.35, 0.32]; 
    upperOrange = [0.07, 0.85, 1.00]; 
    orangeMask = (hsvImage(:,:,1) >= 0.02 & hsvImage(:,:,1) <= 0.07) & ...
                 (hsvImage(:,:,2) >= 0.35 & hsvImage(:,:,2) <= 0.85) & ...
                 (hsvImage(:,:,3) >= 0.32 & hsvImage(:,:,3) <= 1.00);

    % step 8: Merge both segmentations
    finalSegmentedImage = binaryImage | orangeMask;

    % step 9: Apply Morphological operations
    % Close small gaps
    finalSegmentedImage = imclose(finalSegmentedImage,strel('disk',8)); 
    % Remove small noise
    finalSegmentedImage = imopen(finalSegmentedImage,strel('disk',8)); 

    % step 10: Compute Dice Similarity Score (DSc)
    intersection = sum(finalSegmentedImage(:) & gtMask(:));
    diceScore = (2 * intersection) / (sum(finalSegmentedImage(:)) + sum(gtMask(:)) + 1e-6);
    dscScores(i) = diceScore;

    % step 11: Store segmented image, GT mask, and frame name
    segmentedImages{i} = finalSegmentedImage;
    gtMasks{i} = gtMask;
    frameNames{i} = rgbFiles(i).name;
    
end

%% **Display Best and Worst Segmentation Results**
% Sort Dice Scores
[sortedScores, sortedIdx] = sort(dscScores, 'descend');

bestIdx = sortedIdx(1:5);  % Best 5 (highest Dice Scores)
worstIdx = sortedIdx(end-4:end);  % Worst 5 (lowest Dice Scores)

%% **Display Best 5 Segmentations**
for i = 1:5
    figure;
    sgtitle(sprintf('Top Five Best Segmentation Image : %d',i));
    subplot(1,2,1); imshow(segmentedImages{bestIdx(i)});
    title(sprintf(' %s | DSC: %.4f',frameNames{bestIdx(i)},dscScores(bestIdx(i))));
    
    subplot(1,2,2); imshow(gtMasks{bestIdx(i)});
    title(sprintf('Ground Truth: %s_GT', frameNames{bestIdx(i)}));
end

%% **Display Worst 5 Segmentations**
for i = 1:5
    figure;
    sgtitle(sprintf('Top Five Worst Segmentation Image : %d',i));
    subplot(1,2,1); imshow(segmentedImages{worstIdx(i)});
    title(sprintf('%s | DSC: %.4f', frameNames{worstIdx(i)},dscScores(worstIdx(i))));
    
    subplot(1,2,2); imshow(gtMasks{worstIdx(i)});
    title(sprintf('Ground Truth: %s_GT', frameNames{worstIdx(i)}));
end

%% **Dice Score Bar Graph**
meanDSC = mean(dscScores);
stdDSC = std(dscScores);

figure;
bar(dscScores);
xlabel('Frame Index');
ylabel('Dice Similarity Score');
title(sprintf('Mean DSC: %.4f | Standard Deviation: %.4f', meanDSC, stdDSC));
ylim([0 1]); % DSC scores range from 0 to 1

% Save segmented images
for i = 1:length(segmentedImages)
    % Generate filename using frame name (optional) or index
    frameName = sprintf('segmented_%03d.png', i);
    
    % Full path to save the file
    savePath = fullfile(outputFolder, frameName);
    
    % Save image (ensure logical mask is converted to uint8)
    imwrite(uint8(segmentedImages{i}) * 255, savePath);
end

disp("Program run successfully");
