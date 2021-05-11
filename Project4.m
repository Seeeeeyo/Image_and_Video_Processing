%% Introduction to Image and Video Processing 
% coronaproject 4: video processing
%% Task 1
clear; clc; close all; 

% read the running video 
vrun0 = VideoReader('run1.mp4')
% gets 6 frames from this video 
runframe1 = rgb2gray(read(vrun0,104));
runframe2 = rgb2gray(read(vrun0,107));
runframe3 = rgb2gray(read(vrun0,110));
runframe4 = rgb2gray(read(vrun0,113));
runframe5 = rgb2gray(read(vrun0,116));
runframe6 = rgb2gray(read(vrun0,119));
% show those frames
figure;subplot(4,6,1);imshow(runframe1);
subplot(4,6,2);imshow(runframe2);
subplot(4,6,3);imshow(runframe3);
subplot(4,6,4);imshow(runframe4);
subplot(4,6,5);imshow(runframe5);
subplot(4,6,6);imshow(runframe6);
% get the size of a frame 
[hrun wrun] = size(runframe1);
% instanciate the differences' matrices to zeros 
rundiff0 = zeros(hrun,wrun);
rundiff1 = zeros(hrun,wrun);
rundiff2 = zeros(hrun,wrun);
rundiff3 = zeros(hrun,wrun);
rundiff4 = zeros(hrun,wrun);
rundiff5 = zeros(hrun,wrun);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hrun
    for j=1:wrun
        if(abs(runframe2(i,j)-runframe1(i,j)) >= 15)
            rundiff1(i,j) = 1;
        end
        if(abs(runframe3(i,j)-runframe2(i,j)) >= 15)
            rundiff2(i,j) = 1;
        end
        if(abs(runframe4(i,j)-runframe3(i,j)) >= 15)
            rundiff3(i,j) = 1;
        end
        if(abs(runframe5(i,j)-runframe4(i,j)) >= 15)
            rundiff4(i,j) = 1;
        end
        if(abs(runframe6(i,j)-runframe5(i,j)) >= 15)
            rundiff5(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(4,6,7);imshow(rundiff0);
subplot(4,6,8);imshow(rundiff1);
subplot(4,6,9);imshow(rundiff2);
subplot(4,6,10);imshow(rundiff3);
subplot(4,6,11);imshow(rundiff4);
subplot(4,6,12);imshow(rundiff5);
subplot(4,6,13); imshow(rundiff0);
subplot(4,6,19); imshow(rundiff0);

% compute the MEIs using Lukas-Kanade
wsizerun = 5;
[urun0, vrun0] = LucasKanade(runframe1, runframe2, wsizerun);
qrun(:,:,1) = urun0;
qrun(:,:,2) = vrun0;
subplot(4,6,14);imshow(urun0);
subplot(4,6,20);imshow(vrun0);

[urun1, vrun1] = LucasKanade(runframe2, runframe3, wsizerun);
qrun(:,:,1) = urun1;
qrun(:,:,2) = vrun1;
subplot(4,6,15);imshow(urun1);
subplot(4,6,21);imshow(vrun1);

[urun2, vrun2] = LucasKanade(runframe3, runframe4, wsizerun);
qrun(:,:,1) = urun2;
qrun(:,:,2) = vrun2;
subplot(4,6,16);imshow(urun2);
subplot(4,6,22);imshow(vrun2);

[urun3, vrun3] = LucasKanade(runframe4, runframe5, wsizerun);
qrun(:,:,1) = urun3;
qrun(:,:,2) = vrun3;
subplot(4,6,17);imshow(urun3);
subplot(4,6,23);imshow(vrun3);

[urun4, vrun4] = LucasKanade(runframe5, runframe6, wsizerun);
qrun(:,:,1) = urun4;
qrun(:,:,2) = vrun4;
subplot(4,6,18);imshow(urun4);
subplot(4,6,24);imshow(vrun4);
%%
% Bonus 
% add noise 
runnoise1 = imnoise(runframe1,'Gaussian');
runnoise2 = imnoise(runframe2,'Gaussian');
runnoise3 = imnoise(runframe3,'Gaussian');
runnoise4 = imnoise(runframe4,'Gaussian');
runnoise5 = imnoise(runframe5,'Gaussian');
runnoise6 = imnoise(runframe6,'Gaussian');

% show those noisy frames
figure;subplot(4,6,1);imshow(runnoise1);
subplot(4,6,2);imshow(runnoise2);
subplot(4,6,3);imshow(runnoise3);
subplot(4,6,4);imshow(runnoise4);
subplot(4,6,5);imshow(runnoise5);
subplot(4,6,6);imshow(runnoise6);

% get the size of a frame 
[hrunnoise, wrunnoise] = size(runnoise1);
% instanciate the differences' matrices to zeros 
rundiff0n = zeros(hrunnoise,wrunnoise);
rundiff1n = zeros(hrunnoise,wrunnoise);
rundiff2n = zeros(hrunnoise,wrunnoise);
rundiff3n = zeros(hrunnoise,wrunnoise);
rundiff4n = zeros(hrunnoise,wrunnoise);
rundiff5n = zeros(hrunnoise,wrunnoise);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hrunnoise
    for j=1:wrunnoise
        if(abs(runnoise2(i,j)-runnoise1(i,j)) >= 15)
            rundiff1n(i,j) = 1;
        end
        if(abs(runnoise3(i,j)-runnoise2(i,j)) >= 15)
            rundiff2n(i,j) = 1;
        end
        if(abs(runnoise4(i,j)-runnoise3(i,j)) >= 15)
            rundiff3n(i,j) = 1;
        end
        if(abs(runnoise5(i,j)-runnoise4(i,j)) >= 15)
            rundiff4n(i,j) = 1;
        end
        if(abs(runnoise6(i,j)-runnoise5(i,j)) >= 15)
            rundiff5n(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(4,6,7);imshow(rundiff0n);
subplot(4,6,8);imshow(rundiff1n);
subplot(4,6,9);imshow(rundiff2n);
subplot(4,6,10);imshow(rundiff3n);
subplot(4,6,11);imshow(rundiff4n);
subplot(4,6,12);imshow(rundiff5n);
%subplot(4,6,14); imshow(rundiff0n);
% remove the noise
runnoise1nn = medfilt2(runnoise1);
runnoise2nn = medfilt2(runnoise2);
runnoise3nn = medfilt2(runnoise3);
runnoise4nn = medfilt2(runnoise4);
runnoise5nn = medfilt2(runnoise5);
runnoise6nn = medfilt2(runnoise6);

subplot(4,6,13); imshow(runnoise1nn);
subplot(4,6,14); imshow(runnoise2nn);
subplot(4,6,15); imshow(runnoise3nn);
subplot(4,6,16); imshow(runnoise4nn);
subplot(4,6,17); imshow(runnoise5nn);
subplot(4,6,18); imshow(runnoise6nn);

% instanciate the differences' matrices to zeros 
rundiff0nn = zeros(hrunnoise,wrunnoise);
rundiff1nn = zeros(hrunnoise,wrunnoise);
rundiff2nn = zeros(hrunnoise,wrunnoise);
rundiff3nn = zeros(hrunnoise,wrunnoise);
rundiff4nn = zeros(hrunnoise,wrunnoise);
rundiff5nn = zeros(hrunnoise,wrunnoise);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hrunnoise
    for j=1:wrunnoise
        if(abs(runnoise2nn(i,j)-runnoise1nn(i,j)) >= 15)
            rundiff1nn(i,j) = 1;
        end
        if(abs(runnoise3nn(i,j)-runnoise2nn(i,j)) >= 15)
            rundiff2nn(i,j) = 1;
        end
        if(abs(runnoise4nn(i,j)-runnoise3nn(i,j)) >= 15)
            rundiff3nn(i,j) = 1;
        end
        if(abs(runnoise5nn(i,j)-runnoise4nn(i,j)) >= 15)
            rundiff4nn(i,j) = 1;
        end
        if(abs(runnoise6nn(i,j)-runnoise5nn(i,j)) >= 15)
            rundiff5nn(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(4,6,19);imshow(rundiff0nn);
subplot(4,6,20);imshow(rundiff1nn);
subplot(4,6,21);imshow(rundiff2nn);
subplot(4,6,22);imshow(rundiff3nn);
subplot(4,6,23);imshow(rundiff4nn);
subplot(4,6,24);imshow(rundiff5nn);

%% Task 2
%close all; 
% creates a morphological structuring element (disk-shapped with radius 7)
s1  =  strel('disk',2);
% perfom the opening
open1 = imopen(rundiff1,s1); 
open2 = imopen(rundiff2,s1); 
open3 = imopen(rundiff3,s1); 
open4 = imopen(rundiff4,s1); 
open5 = imopen(rundiff5,s1); 

figure;subplot(1,5,1);imshow(open1);
subplot(1,5,2);imshow(open2);
subplot(1,5,3);imshow(open3);
subplot(1,5,4);imshow(open4);
subplot(1,5,5);imshow(open5);
% figure; imshow(rundiff1);
% figure; imshow(rundiff2);
% figure; imshow(rundiff3);
% figure; imshow(rundiff4);
% figure; imshow(rundiff5);

%% Task 3
% detect the edges with the Canny method
edgerun1 = edge(open1,'canny');
edgerun2 = edge(open2,'canny');
edgerun3 = edge(open3,'canny');
edgerun4 = edge(open4,'canny');
edgerun5 = edge(open5,'canny');

figure;subplot(1,5,1);imshow(edgerun1);
subplot(1,5,2);imshow(edgerun2);
subplot(1,5,3);imshow(edgerun3);
subplot(1,5,4);imshow(edgerun4);
subplot(1,5,5);imshow(edgerun5);




%% VIDEO 2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%close all; 
% read the boxing video 
vhand = VideoReader('hand1.mp4')
% gets 2 frames from this video 
handframe1 = rgb2gray(read(vhand,75));
handframe2 = rgb2gray(read(vhand,85));

% show those frames
figure;subplot(3,3,1);imshow(handframe1);
subplot(3,3,2);imshow(handframe2);

% get the size of a frame 
[hbox, wbox] = size(handframe1);
% instanciate the differences' matrices to zeros 
handdiff1 = zeros(hbox,wbox);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hbox
    for j=1:wbox
        if(abs(handframe2(i,j)-handframe1(i,j)) >= 40)
            handdiff1(i,j) = 1;
        end 
    end
end
% show the MEIs
subplot(3,3,3);imshow(handdiff1); 

% compute the MEIs using Lukas-Kanade
wsizehand = 10;
[uhand0, vhand0] = LucasKanade(handframe2, handframe1, wsizehand);
qhand(:,:,1) = urun0;
qhand(:,:,2) = vrun0;
subplot(3,3,4);imshow(uhand0);
subplot(3,3,5);imshow(vhand0);
%%
% Bonus 
% add noise 
handnoise1 = imnoise(handframe1,'Gaussian');
handnoise2 = imnoise(handframe2,'Gaussian');
% show those noisy frames
figure;
% subplot(3,3,6);imshow(handnoise1);
% subplot(3,3,7);imshow(handnoise2);
subplot(3,3,1);imshow(handnoise1);
subplot(3,3,2);imshow(handnoise2);

% get the size of a frame 
[hhandnoise, whandnoise] = size(handnoise2);
% instanciate the differences' matrices to zeros 
handdiff0n = zeros(hhandnoise,whandnoise);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hhandnoise
    for j=1:whandnoise
        if(abs(handnoise2(i,j)-handnoise1(i,j)) >= 40)
            handdiff0n(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(3,3,3);imshow(handdiff0n);
% remove the noise
handnoise1nn = medfilt2(handnoise1);
handnoise2nn = medfilt2(handnoise2);
subplot(3,3,5);imshow(handnoise1nn);
subplot(3,3,6);imshow(handnoise2nn);

% instanciate the differences' matrices to zeros 
handdiff1nn = zeros(hhandnoise,whandnoise);

% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hhandnoise
    for j=1:whandnoise
        if(abs(handnoise2nn(i,j)-handnoise1nn(i,j)) >= 40)
            handdiff1nn(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(3,3,7);imshow(handdiff1nn);

%% Task 2
%close all; 
% creates a morphological structuring element
s1  =  strel('disk',1);
% perfom the opening
openhand1 = imopen(handdiff1,s1); 

figure;subplot(2,3,1);imshow(openhand1);
% Task 3
edgehand1 = edge(openhand1,'canny');
subplot(2,3,2);imshow(edgehand1);












%% VIDEO 3 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%close all; 
% read the boxing video 
vwalk = VideoReader('walk1.mp4')
% gets 2 frames from this video 
walkframe1 = rgb2gray(read(vwalk,305));
walkframe2 = rgb2gray(read(vwalk,319));

% show those frames
figure;subplot(3,3,1);imshow(walkframe1);
subplot(3,3,2);imshow(walkframe2);

% get the size of a frame 
[hwalk, wwalk] = size(walkframe1);
% instanciate the differences' matrices to zeros 
walkdiff1 = zeros(hwalk,wwalk);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hwalk
    for j=1:wwalk
        if(abs(walkframe2(i,j)-walkframe1(i,j)) >= 50)
            walkdiff1(i,j) = 1;
        end 
    end
end
% show the MEIs
subplot(3,3,3);imshow(walkdiff1); 

% compute the MEIs using Lukas-Kanade
wsizehand = 20;
[uwalk0, vwalk0] = LucasKanade(walkframe1, walkframe2, wsizehand);
qwalk(:,:,1) = uwalk0;
qwalk(:,:,2) = vwalk0;
subplot(3,3,4);imshow(uwalk0);
subplot(3,3,5);imshow(vwalk0);
%%
% Bonus 
% add noise 
walknoise1 = imnoise(walkframe1,'salt & pepper');
walknoise2 = imnoise(walkframe2,'salt & pepper');
% show those noisy frames
figure; 
subplot(3,3,1);imshow(walknoise1);
subplot(3,3,2);imshow(walknoise2);

% get the size of a frame 
[hwalknoise1, wwalknoise1] = size(walknoise1);
% instanciate the differences' matrices to zeros 
walkdiff0n = zeros(hwalknoise1,wwalknoise1);
% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hwalknoise1
    for j=1:wwalknoise1
        if(abs(walknoise2(i,j)-walknoise1(i,j)) >= 50)
            walkdiff0n(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(3,3,4);imshow(walkdiff0n);
% remove the noise
walknoise1nn = medfilt2(walknoise1);
walknoise2nn = medfilt2(walknoise2);

subplot(3,3,5);imshow(walknoise1nn);
subplot(3,3,6);imshow(walknoise2nn);

% instanciate the differences' matrices to zeros 
walkdiff1nn = zeros(hwalknoise1,wwalknoise1);

% compute the MEIs using inter-frame differences D(x, y, t) = I(x, y, t)−I(x, y, t−1)
for i=1:hwalknoise1
    for j=1:wwalknoise1
        if(abs(walknoise2nn(i,j)-walknoise1nn(i,j)) >= 50)
            walkdiff1nn(i,j) = 1;
        end
    end
end
% show the MEIs
subplot(3,3,9);imshow(walkdiff1nn);

%% Task 2
%close all; 
% creates a morphological structuring element
s1  =  strel('disk',2);
% perfom the opening
openwalk1 = imopen(walkdiff1,s1); 

figure;subplot(2,3,1);imshow(openwalk1);
% Task 3
edgewalk1 = edge(openwalk1,'canny');
subplot(2,3,2);imshow(edgewalk1);
%% Task 4

% Extract the shape descriptor for the MEI outlines of the actions using Hu moments
humrun1 = hu_moments(edgerun1);
humrun2 = hu_moments(edgerun2);
humrun3 = hu_moments(edgerun3);
humrun4 = hu_moments(edgerun4);
humrun5 = hu_moments(edgerun5);

humhand1 = hu_moments(edgehand1);
humwalk1 = hu_moments(edgewalk1);

%% Task 5 
% mean square error to compare the shape descriptors of the different
% actions 
mse1 = immse(humrun2,humhand1);
mse2 = immse(humwalk1,humhand1);
mse3 = immse(humrun2,humwalk1);
mse4 = immse(humrun1,humrun2);
mse5 = immse(humrun2,humrun3);