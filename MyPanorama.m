function [pano] = MyPanorama()

%% YOUR CODE HERE.
% Must load images from ../Images/Input/
% Must return the finished panorama.
sprintf('Hi there!')

jpgFiles = dir('../Images/input/*.jpg')

transforms = projective2d.empty(1, 0);
transforms(1) = projective2d(eye(3));

for i = 2:length(jpgFiles)
    disp(jpgFiles(i).name)
    I1 = imread(strcat('../Images/input/', jpgFiles(i - 1).name));
    I2 = imread(strcat('../Images/input/', jpgFiles(i).name));
    
    Ig1 = rgb2gray(I1);
    Ig2 = rgb2gray(I2);
    
    c1 = cornermetric(Ig1);
    c2 = cornermetric(Ig2);
    
    Nbest1 = ANMS(c1);
    Nbest2 = ANMS(c2);
    
%     figure(i)
%     imshow(I1)
%     hold on
%     plot(Nbest1(2,:), Nbest1(1,:), '.')
%     hold off
%     
%     figure(i + 1)
%     imshow(I2)
%     hold on
%     plot(Nbest2(2,:), Nbest2(1,:), '.')
%     hold off
    
    [fDesc1, fCoords1] = getFeatureDescriptors(Nbest1, Ig1);
    [fDesc2, fCoords2] = getFeatureDescriptors(Nbest2, Ig2);
    
    matchedFeatureIndices = getMatchedFeatures(fDesc1, fDesc2);
    
    matchedPoints1 = fCoords1(:, matchedFeatureIndices(:,1));
    matchedPoints2 = fCoords2(:, matchedFeatureIndices(:,2));
    % Flip rows to create x,y coordinates instead of matrix coords
    matchedPoints1 = [matchedPoints1(2,:);
                      matchedPoints1(1,:)]';
    matchedPoints2 = [matchedPoints2(2,:);
                      matchedPoints2(1,:)]';
                  
%     showMatchedFeatures(Ig1, Ig2, matchedPoints1, matchedPoints2, 'montage')

    homoMatrix = RANSAC(matchedPoints1, matchedPoints2, 100, Ig1, Ig2);
    
    homoProj = invert(projective2d(homoMatrix'));
    
    transforms(1, i) = homoProj;
    transforms(1, i).T = transforms(1, i).T * transforms(1, i - 1).T;
    
end

xlim = zeros(length(jpgFiles), 2)
ylim = zeros(length(jpgFiles), 2)

I1 = rgb2gray(imread('../Images/input/1.jpg'));

maxSizeX = size(I1,2);
maxSizeY = size(I1,1);

for i = 1:length(jpgFiles)
    I = imread(strcat('../Images/input/', jpgFiles(i).name));
    [xlim(i, :), ylim(i, :)] = outputLimits(transforms(i), [1 size(I, 2)], [1 size(I, 1)]);
    
    if size(I,2) > maxSizeX
        maxSizeX = size(I,2);
    end
    
    if size(I,1) > maxSizeY
        maxSizeY = size(I,1);
    end
end

minX = min([1; xlim(:)]);
minY = min([1; ylim(:)]);

maxX = max([maxSizeX; xlim(:)]);
maxY = max([maxSizeY; ylim(:)]);

xLimits = [minX maxX];
yLimits = [minY maxY];

width = round(maxX - minX);
height = round(maxY - minY);

panorama = zeros(height, width, 'like', I1);
% panorama = imresize(panorama, .2);
panoramaRef = imref2d([size(panorama,1) size(panorama,2)], xLimits, yLimits);


for i = 1:length(jpgFiles)
    Ig = rgb2gray(imread(strcat('../Images/input/', jpgFiles(i).name)));
    warpIm = imwarp(Ig, transforms(i), 'OutputView', panoramaRef);
    panorama = imfuse(panorama, panoramaRef, warpIm, panoramaRef, 'blend');
end

end

function Nbest = ANMS(corners)
    rMax = imregionalmax(corners)
    
    localMaxInd = find(rMax);
    
    [rows columns] = ind2sub(size(corners), localMaxInd);
    
    Nstrong = length(localMaxInd);
    
    r = Inf([1 Nstrong]);
    
    for i = 1:Nstrong
        for j = 1:Nstrong
            if corners(rows(j), columns(j)) > corners(rows(i), columns(i))
                ED = (rows(j) - rows(i))^2 + (columns(j) - columns(i))^2;
                
                if ED < r(i)
                    r(i) = ED;
                end
            end
        end
    end
    
    [V, ind] = maxk(r, 300);
    Nbest = [transpose(rows(ind));
              transpose(columns(ind))]
end

function [featDescriptors, featCoords] = getFeatureDescriptors(Nbest, I)
    width = length(I(1,:));
    height = length(I(:, 1));
    filter = fspecial('gaussian', 40);
    interInd = 1:25:1600;
    featDescriptors = zeros(length(Nbest(1,:)), 64);
    for i = 1:length(Nbest)
        row = Nbest(1, i);
        column = Nbest(2, i);
        featureVec = zeros(1,64);
        
        if ~(column < 20 || column > width - 20 || row < 20 || row > height - 20)
            subImage = I(row-19:row+20, column-19:column+20);
            subBlur = imfilter(subImage, filter);
            interSubImage = double(subBlur(interInd))
            featureVec = double(interSubImage);
            avg = mean(featureVec);
            dev = std(featureVec);
            featureVec = featureVec-avg;
            featureVec = featureVec/dev;
        end
        
        featDescriptors(i, :) = featureVec;
    end
    
    availableFeats = any(featDescriptors,2);
    featCoords = Nbest(:, availableFeats');
    featDescriptors = featDescriptors(availableFeats, :);
end

function matchedFeatureIndexes = getMatchedFeatures(fDesc1, fDesc2)
    numReps = length(fDesc2(:,1));
    matchedFeatureIndexes = zeros(length(fDesc1(:, 1)), 2);
    
    for i = 1:length(fDesc1(:,1))
        subtractMatrix = repmat(fDesc1(i,:), numReps, 1);
        diffMatrix = fDesc2 - subtractMatrix;
        sqrdDiffMatrix = diffMatrix.^2;
        ssdVec = sum(sqrdDiffMatrix, 2);
        [twoClosest,Ind] = mink(ssdVec, 2);
        closeRatio = twoClosest(1)/twoClosest(2);
        if closeRatio < 0.5
            matchedFeatureIndexes(i,:) = [i, Ind(1)];
        else
            matchedFeatureIndexes(i,:) = [0 0];
        end
    end
    
    matchedFeatureIndexes = matchedFeatureIndexes(any(matchedFeatureIndexes, 2), :);
end

function homoMatrix = RANSAC(matchingPoints1, matchingPoints2, N, Ig1, Ig2)
    i = 1;
    percentInliers = 0;
    inlierMask = zeros(length(matchingPoints1(:,1)));

    while i < N + 1 && percentInliers < .9
        hIndices = randi([1 length(matchingPoints1(:,1))], 1, 4);
        estHomoPoints1 = matchingPoints1(hIndices,:);
        estHomoPoints2 = matchingPoints2(hIndices,:);
        
        H = est_homography(estHomoPoints2(:,1), estHomoPoints2(:,2), estHomoPoints1(:,1), estHomoPoints1(:,2));
        [X,Y] = apply_homography(H, matchingPoints1(:,1), matchingPoints1(:,2));
        estHomoCoords = [X Y];
        
        SSD = sum((matchingPoints2 - estHomoCoords).^2, 2);
        tempInlierMask = SSD < 400;
        tempPercentInliers = sum(tempInlierMask)/length(matchingPoints1(:,1));
        if tempPercentInliers > percentInliers
            inlierMask = tempInlierMask;
            percentInliers = tempPercentInliers;
        end
        i = i + 1;
    end
    
    inlierSet1 = matchingPoints1(inlierMask, :);
    inlierSet2 = matchingPoints2(inlierMask, :);
    
    showMatchedFeatures(Ig1, Ig2, inlierSet1, inlierSet2, 'montage');
    
    homoMatrix = est_homography(inlierSet2(:,1), inlierSet2(:,2), inlierSet1(:,1), inlierSet1(:, 2));
    [X,Y] = apply_homography(H, matchingPoints1(:,1), matchingPoints1(:,2));
    testHomo = [X Y];
end
