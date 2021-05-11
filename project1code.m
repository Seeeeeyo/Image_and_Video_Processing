%% Project 1
% Selim Gilon - i 6192074

%% Question 1 
%% 1.1.a

close all; clear all; clc
%x = imread('circlessmall.png');
x = imread('building.jpeg');

%converts the colored image to a grayscale
xd = double(rgb2gray(x));
%get the size of the matrix = #pixels of the image 
[height width] = size(xd);

%Sobel x kernel for vertical edge detection
kernelx = [-1 0 1;-2 0 2;-1 0 1]; 
%Sobel y kernel for horizontal edge detection
kernely = [-1 -2 -1;0 0 0;1 2 1]; 
%Sobel kernel for diagonal edge detection
kernelxy = [-2 -1 0; -1 0 1; 0 1 2];

y1 = imfilter(xd,kernelx);
y2 = imfilter(xd,kernely);
y12 = imfilter(xd,kernelxy);
% Y5 = y1+y2;

% figure; imshow(Y5/255);
figure;subplot(3,3,1);imshow(xd/255); title('original image');
subplot(3,3,2);imshow(y1/255); title('Sobel vertical edges (matrix input)');
subplot(3,3,3);imshow(y2/255); title('sobel horizontal edges (matrix input)');
subplot(3,3,4);imshow(y12/255); title('sobel diagonal edges(matrix input)');

%  figure;imshow(xd/255); title('original image');
%  figure;imshow(y1/255); title('Sobel vertical edges (matrix input)');
%  figure;imshow(y2/255); title('sobel horizontal edges (matrix input)');
%  figure;imshow(y12/255); title('sobel diagonal edges(matrix input)');


% using the edge detection Sobel from Matlab 
y3 = edge(xd,'sobel');
subplot(3,3,5);imshow(y3); title('Sobel built-in filter');
% figure;imshow(y3); title('Sobel built-in filter');

%vertical manual edge detection
for i=2:(height-1)
    for j=2:(width-1)
        matrixv = xd(i-1:i+1,j-1:j+1) .* kernelx;
        s=sum(matrixv(:)); 
        xv(i-1, j-1)=s;
    end
end
% xv
 subplot(3,3,6);imshow(xv/255); title('Sobel manual vertical edges');
% figure; imshow(xv/255); title('Sobel manual vertical edges');

%horizontal manual edge detection
for i=2:(height-1)
    for j=2:(width-1)
        matrixh = xd(i-1:i+1,j-1:j+1) .* kernely;
        s=sum(matrixh(:)); 
        xh(i-1, j-1)=s;
    end
end
% xh
 subplot(3,3,7);imshow(xh/255); title('sobel manual horizontal edges');
% figure;imshow(xh/255); title('sobel manual horizontal edges');
 

%horizontal and vertical manual edge detection (square root of the sum of
%the squared)
for i=2:(height-1)
    for j=2:(width-1)
        matrix1 = xd(i-1:i+1,j-1:j+1) .* kernelx;
        matrix2 = xd(i-1:i+1,j-1:j+1) .* kernely;
        s1=sum(matrix1(:)); 
        s2=sum(matrix2(:));
        s = sqrt(s1^2 + s2^2);
        xb(i-1, j-1)=s;
    end
end
% xb
 subplot(3,3,8);imshow(xb/255); title('Sobel manual edges');
%figure;imshow(xb/255); title('Sobel manual diagonal edges');

%% 1.1.b
% Using the built in Canny filter 
y4 = edge(xd,'canny');
figure;imshow(y4); title('Canny built-in filter');

% start of the manual canny but didn't finish
% blur using gaussian smoothing filter 
% imageBlur = imgaussfilt(xd,1.2);
% figure;imshow(imageBlur/255); title('automatic gaussian smoothing filter');
% 
% K = zeros(height-2, width-2);
% for i=2:(height-1)
%     for j=2:(width-1)
%         m1 = xd(i-1:i+1,j-1:j+1) .* kernelx;
%         m2 = xd(i-1:i+1,j-1:j+1) .* kernely;
%         ss1=sum(matrix1(:)); 
%         ss2=sum(matrix2(:));
%         ss = ss1 + ss2;
%         K(i-1, j-1)=ss;
%     end
% end
% figure;imshow(xb/255); title('Gradient calculation using Sobel for Canny');

%% 1.2
% 1.2.a
close all; clear all; clc
% x = imread('buildingbw.jpeg');
 x = imread('circlessmallbw.jpeg');

% xnoisy = imnoise(x, 'salt & pepper', 0.0005);
xnoisy = imnoise(x, 'gaussian');
figure; subplot(2,2,1);imshow(xnoisy); title('noisy image');

% other way to add noise manually
% xn = xd;
% r=rand(size(xn));
% density=.05;
% r(r<density)=255;
% xn=uint8(r+ouble(xn));
% figure; imshow(xn); title('noisy image')

% using built in sobel
sobel_noisy = edge(xnoisy,'sobel');
subplot(2,2,2); imshow(sobel_noisy); title('sobel with noisy image')

% using laplacian 
l = fspecial('laplacian');
lap_noisy = imfilter(xnoisy, l);
subplot(2,2,3);imshow(lap_noisy); title('Laplacian noisy image')

% 1.2.b
% using laplacian of gaussian
logl = edge(xnoisy,'log'); 
subplot(2,2,4);imshow(logl); title('LoG with noisy image')

%% Question 2
%% 2.2.1
close all; clear all; clc
x = imread('buildingbw.jpeg');
% x = imread('circlesmallbw.png');

[height,width] = size(x);

% Try of manual cartesian to polar - didn't succeed
% for r=0:0.01:(height/2)
%   for q=0:0.01:(2*pi)
%  
%   polarX = (r * cos(q)) + width/2;
%   polarY = (r * sin(q)) + height/2;
% 
%    x = (1-q/2*pi)*width;
% 
%       outputIndex = (x + r) * width
%       inputIndex = (polarX + polarY) * width;
% 
%       out(outputIndex) = x(inputIndex);
%   end
% end
% figure; imshow(out)

% calculate the new coordinates 
[xcoor,ycoor] = meshgrid((1:height), (1:width));
theta = atan2(ycoor,xcoor);
rho = hypot(xcoor,ycoor);
zerosm = zeros(size(theta));

% Tried with interp2 but I got a weird error
% vq = interp2(xcoo, ycoo,x,theta, rho);
% figure;subplot(1,2,1); imshow(x);
% subplot(1,2,2); surf(xcoo,ycoo,vq); view(2);

figure;subplot(1,2,1); imshow(x);
subplot(1,2,2); warp(theta,rho,zerosm, x); view(2);

%%
%% 2.2.3.1
close all; clear all; clc
x = imread('me.jpg');
[height,width] = size(x);

for i=1:height
    for j=1:width
        offsetx = 0; 
        offsety = (30*sin(2*pi*j/150));
        if offsety+i < width
            output(i,j) = x(mod(round((i+offsety),0),height)+1,j);
        else 
            output(i,j) = 0;
        end    
    end
end

figure;subplot(1,2,1); imshow(x);
subplot(1,2,2); imshow(output);
%%
%%  2.2.3.2
close all; clear all; clc
x = imread('me.jpg');
[height,width] = size(x);

for i=1:height
    for j=1:width
        offsetx = 100 * sin(2*pi*i/150);
        offsety = 100 * cos(2*pi*j/150);
        if offsety+i < height && offsetx+j < width
            output(i,j) = x(mod(round((i+offsety),0),height)+1, mod(round((j+offsetx),0),width)+1);
        else 
            output(i,j) = 0;
        end    
    end
end
figure;subplot(1,2,1); imshow(x);
subplot(1,2,2); imshow(output);
%% 2.2.3.3
close all; clear all; clc
x = imread('me.jpg');
[height,width] = size(x);

for i=1:height
    for j=1:width
        offsetx = round(200 * sin(10*pi*i/(10*width)),0);
        offsety = 0;
        if offsetx+j < width 
           
            output(i,j) = x(i, mod(j+offsetx,width)+1);
        else 
            output(i,j) = 0;
        end    
    end
end

figure;subplot(1,2,1); imshow(x);
subplot(1,2,2); imshow(output);
