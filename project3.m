%% Introduction to Image and Video Processing - coronaproject 3: compression, morphological image processing
%% Lab 3
%% Task 1 - Hoffman 
clc; close all; clear

% my last name is Gilon 
lastName = 'GILON'; 
% length of my last name 
l = length(lastName);

% count the number of occurences of each letters 
numberOfG = count(lastName,'G');
numberOfI = count(lastName,'I');
numberOfL = count(lastName,'L');
numberOfO = count(lastName,'O');
numberOfN = count(lastName,'N');

% calculate the probabilty of each letter 
probaG = numberOfG/l;
probaI = numberOfI/l;
probaL = numberOfL/l;
probaO = numberOfO/l;
probaN = numberOfN/l;

% put all the proba into an array
probs = [probaG probaI probaL probaO probaN]; 
% sort the proba in descending order 
probsSorted = sort(probs,'descend');

% sourceReduction is a matrix with the data fro each source reudction steps
sourceReduction = 0;

% add the original probablities to the source reduction matrix 
for j=1:l
    sourceReduction(1,j) = probsSorted(j); 
end

% print the source reduction at the start if needed
sourceReduction;

% number of columns in the source reduction matrix 
stepNeeded = round(l/2,0)+1;

% counter 1
count = 1;
% Counter 2
c = 3;
% counter 3
s = 0; 

% Creation of the 1st step of the algo: the source reduction
% i starts at 2 because the first column of the source reduction is the
% original probas
for i=2:stepNeeded
    tmp1 = sourceReduction(count,l-s);  % get the last non 0 element of the specific column 
    tmp2 = sourceReduction(count,l-(s+1)); % get the second to last non 0 element of the specific column 
    parent = tmp1 + tmp2; % sum of the 2 smallest probas for the next column 
    arrayParent(count) = parent; % stores the sum of each iteration 
    arrayChildren(1,count) = tmp2; % stores the children of each iteration 
    arrayChildren(2,count) = tmp1; % stores the children of each iteration 
    rest = sourceReduction(count,1:c); % the rest is all the elements of the column besides the 2 smallest ones 
    tmpArray = [parent rest]; % the tmp array is the array of values of the next iteration 
    tmpArrayS = sort(tmpArray,'descend'); % Sorts the array to have the values in the right order (decreasing) in the tableau
    fullTmpArray = tmpArrayS; % just used to debug
    fullTmpArray((l-s):5) = 0; % assign values to the the last(s) element(s) of the column in order to have number everywhere in the matrix. It is not mandatory, just my preference. 
    sourceReduction(count+1,:) = fullTmpArray; % Add this column of sprted probas in the source reduction matrix 
    count = count + 1; % increment the counter 1 by one 
    s = s + 1; % increment the counter 3 by one 
    c = c - 1; % decrement the counter 2 by one 
end 


arrayChildren; % Used to debug, this is the array of all the "children", the probas that have been added during the source reduction step
arrayParent; % Used to debug, this is the array of all the "parents", the results of the proba that have been added during the source reduction step
tree = [arrayChildren]; % add the children to the tree matrix
sourceReduction = transpose(sourceReduction) % transpose the source reduction matrix to have it as we studied it in class (each source reduction step corresponds to a column of the matrix)
tree(1,4) = sourceReduction(1,4); % add the last elements of the reduction to the tree
tree(2,4) = sourceReduction(2,4); % add the last elements of the reduction to the tree
tree; % used to debug 

% counter for the next loop
count = 1;

% Step 2: source coding 
% j represent the column number of the matrix 
for j=4:-1:1
    % encode the 2 elements of the last source reduction to "0" and "1"
    % where the first one is the biggest proba and is encoded as 0.
    if(j == 4)
       partBit(1,j) = "0";
       partBit(2,j) = "1"; 
       % set the elements that "formed" the proba of the next column (source
       % reduction iteration) to the code of this result + 0 or 1 based on
       % the descending order previously discussed.
    else
        tmp1 = partBit(1,j+1);
        partBit(count+1,j) = append(tmp1,"0");
        partBit(count+2,j) = append(tmp1,"1");
        % get the code of the correct probas of the next column (that do not add) and copy it
        % in the current column at the right spot 
        if(j == 3) % column 3 so only 1 proba to encode, to get from the next column
            tm1 = partBit(2,j+1); % get the code in the next column
            partBit(1,j) = tm1; % set it 
        elseif(j == 2) % column 2 so 2 probas to encode, to get from the next column
            tm21 = partBit(2,j+1); % get the code in the next column
            tm22 = partBit(3,j+1); % get the code in the next column
            partBit(1,j) = tm21; % set it 
            partBit(2,j) = tm22; % set it 
        elseif(j == 1) % column 1 so 3 probas to encode, to get from the next column
            tm31 = partBit(2,j+1); % get the code in the next column
            tm32 = partBit(3,j+1); % get the code in the next column
            tm33 = partBit(4,j+1); % get the code in the next column
            partBit(1,j) = tm31; % set it 
            partBit(2,j) = tm32; % set it 
            partBit(3,j) = tm33; % set it 
        end
        count = count + 1; % increment counter by one.
    end
end
% partBit matrix is the matrix containing the codes 
partBit % print the partBit matrix 

lastNameBits = ""; % instanciation 
finalBits = partBit(:,1) % get and prints the bits of the original probas (letters) 

% append the codes in order to create one code describing my last name 
for i=1:l
    tmp = finalBits(i,1);
    lastNameBits = append(lastNameBits,tmp);
end

lastNameBits % print my encoded last name 

%% Task 2

%% 1
clc; close all; clear; 

x = imread('im2.jpg');
xd  =  double(rgb2gray(x));

%figure;subplot(2,2,1);imshow(xd/255);
% imb = im2bw(x,0.8);
% subplot(2,2,2); imshow(imb);

% compute the mean of the image (in grayscale) 
m = mean(xd(:)); 
% compute its standard deviation 
stdv = std(xd(:)); 
% gets its size 
[h w] = size(xd);
% compute the treshold 
t8 = (m + 0.8*stdv);
% creates a matrix full of zeroes 
z = zeros(h,w);
% show the orginal image 
figure;subplot(3,2,1); imshow(x);title('original image')

% set the values to 1 if they are above the treshold (determined by the
% mean, the standard deviation and its coefficient)
for i=1:h
    for j=1:w
        if(xd(i,j) > (m + 0.8*stdv))
            z(i,j) = 1; 
        end
    end
end

subplot(3,2,2);imshow(z);title('binarized image'); 

%% 2
% creates a morphological structuring element (disk-shapped with radius 7)
s1  =  strel('disk',7);
% perfom the closing - just etra experience 
x1  =  imclose(z,s1);
% perfom the dilation - just etra experience 
x2 = imdilate(z,s1);
% perfom the erosion 
x3 = imerode(z,s1);
% perfom the opening
x4 = imopen(z,s1); 

% show the results 
subplot(3,2,3);imshow(x1);title('closing')
subplot(3,2,4);imshow(x2);title('dilation')
subplot(3,2,5);imshow(x3);title('erosion')
subplot(3,2,6);imshow(x4);title('opening')
figure; imshow(z); title('binarized');
figure;imshow(x1);title('closing')
figure;imshow(x2);title('dilation')
figure;imshow(x3);title('erosion')
figure;imshow(x4);title('opening')

%% Bonus 
close all; clear all; clc

% convert the colored image to gray scale 
img = double(rgb2gray(imread('mint.jpg')));
% show the gray scale image 
figure;imshow(img/255);title('Original mint image');

% the inital sum is just the sum of all gray scale of the matrix of the image
init_sum=sum(img(:));
% the j is is the radius size of the disk-shaped structural element 
% we'll apply this structural element to the orginial gray scale image 
% we start with a radius of size 3, increment it by 5 and the max (last)
% radius size is 50. 
for j = 3:5:50
    % apply an opening with a disk-shaped structural element of radius j 
    b = imopen(img, strel('disk', j));
    % show the image and the radius of its structural element 
    figure;imshow(b/255);title(['Radius:',int2str(j)]);
    % compute the frequency by computing the ratio between the sum of the gray scale levels of the new image
    % and the initial sum which was the sum of the gray scale levels of the
    % original image. Stores this frequency into an array. 
    freq(j)=sum(b(:))/init_sum;
end

% display (plot) the frequencies 
figure;stem(freq);title('Frequency Diagram');
grid on;

