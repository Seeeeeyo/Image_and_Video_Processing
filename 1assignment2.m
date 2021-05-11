%% Task 1
%% 1 
close all; clear all; clc; 
fcx = 4;  % frequency               
fsx = 400; fsy = 400; % sizes                    
dt1 = 1/fsx;  % interval of time x coordinate 
dt2 = 1/fsy;  % interval of time y coordinate
StopTime = 1; % time in seconds
[t1, t2] = ndgrid(0:dt1:StopTime-dt1, 0:dt2:StopTime-dt2);
x = cos(2*pi*(fcx*t1 + fcx*t2));
% visualization
figure;imagesc(x); axis off; 


%% a - on paper 

%% b
clc

f = fft2(x); 
fs = fftshift(f);
fphase = angle(fs);
fmag = abs(fs); 
fpower = fmag.^2;


figure;subplot(2,2,1);imshow(fmag);title('magnitude');
subplot(2,2,2);imshow(fphase);title('Phase angle spectrum');
%subplot(2,2,3);imshow(log(fpower)/25);title('Power spectrum');
subplot(2,2,3);plot(10*log(fpower)/255); title('power spectrum 1D');
subplot(2,2,4);imagesc(x);title('original signal');

%% 2
%% a
%clear; clc; close all; 
%Read the image 
im = imread('brick.jpg');
% Convert it to grayscale 
img = rgb2gray(im);
% show the original image
% subplot(2,2,1);imshow(img);title('original image')

f = fft2(img); 
fs = fftshift(f);
fphase = angle(fs);
fmag = abs(fs);
fpower = fmag.^2; 

figure;subplot(2,2,1);imshow(log(fmag)/25);title('Magnitude');
subplot(2,2,2);imshow(log(fphase));title('Phase angle');
%subplot(2,2,3);imshow(log(abs(fs))/25);title('Fourrier spectrum');
subplot(2,2,3); plot(10*log(fpower)/255); title('power spectrum 1D');
subplot(2,2,4);imshow(img);title('original image');

%% b 
% remove the stongest frequency
fmag2 = fmag; 
[m,i] = max(fmag2);
[mm, ii] = max(m);
iii = i(ii);
fmag2(iii,ii) = 0;


fpower2 = fmag2.^2;
fphase2 = angle(fmag2);
out = ifftshift(fmag2);
out = ifft2(out); 

% n = notchfilter(2,400,400);
% fpower2 = abs(n).^2;
% ou = fs.*n; 
% ou = ifftshift(ou);
% ou = ifft2(ou); 

figure;subplot(2,2,1);imshow(log(fmag2)/25);title('magnitude');
subplot(2,2,2);imshow(fphase);title('Phase angle spectrum');
subplot(2,2,3);plot(10*log(fpower2)/255); title('power spectrum 1D');
subplot(2,2,4);imshow(out);title('image with highest fr removed');


%% Task 2 - Periodic noise removal 
%% 1
close all; clear all; clc; 

%Read the image 
im = imread('deer.jpg');
% Convert it to grayscale 
img = rgb2gray(im);
% show the original image
subplot(2,2,1);imshow(img);title('original image')
% get the size of the image
[h w] = size(img);
% Calculate FFT2 of the image and center it with fftshift
f = fftshift(fft2(img)); 
fr = 40; % frequency of the sinusoidal noise 

% Create a matrix of periodic noise using the sinus function  
n1 = pnoise(fr,h,w,40)
fn = fftshift(fft2(n1));

% add noise to the original image by adding summing the elements of each
% matrix 
for i=1:h
    for j=1:w
        f1(i,j) = im(i,j) + n1(i,j); 
    end
end

% show the noisy image 
subplot(2,2,2);imshow(f1);title('noisy image')
%% extra ------ for me ------ 
% calculate the 2D FFT of the degraded image using fft2
fono2 = fft2(img);
% center it with fftshift
fonos2 = fftshift(fono2); 
% display the noisy image's power spectrum
ps2 = abs(fonos2).^2;
figure;subplot(2,2,1);imshow(10*log(ps2)/255);title('power spectrum 2D');
subplot(2,2,2);plot(10*log(ps2)/255); title('power spectrum 1D');
subplot(2,2,3);mesh(10*log(ps2)/255);title('power spectrum 3D');
subplot(2,2,4);imshow(img);title('original image');

%% 2

% calculate the 2D FFT of the noisy image using fft2
fono = fft2(f1);

fo = fft(f1)
% center it with fftshift
fonos = fftshift(fono); 


% display the noisy image's power spectrum
ps = abs(fonos).^2;

figure;subplot(2,2,1);imshow(10*log(ps)/255);title('power spectrum 2D');
subplot(2,2,2);plot(10*log(ps)/255); title('power spectrum 1D');
subplot(2,2,3);mesh(10*log(ps)/255);title('power spectrum 3D');
%subplot(2,2,4); imshow(fonos/255)
%subplot(2,2,4);imshow(10*log(ps1)/255);
% WHAT DOES IT REVEAL?

%% 3 

% nr = notchfilter(2,h,w)
%no = fft2(nr); 
%nos = fftshift(no); 

 nr = Bnotchfilter(40, h,w,2,30); 
% nr = gaussLPF(10,h,w)

g1 = fonos.*nr; 
% ff3 = fftshift(fft2(Bnotchfilter(fr, h, w, 2, 5)));

y1 = ifftshift(g1);
y1 = ifft2(y1); 

figure;subplot(3,2,1);imshow(y1/255);title('recovered image')
subplot(3,2,2); imshow(nr);title('butterworth notch filter');

% display the noisy image's power spectrum
psg = abs(g1).^2;

subplot(3,2,3);imshow(10*log(psg)/255);title('power spectrum 2D');
subplot(3,2,4);plot(10*log(psg)/255); title('power spectrum 1D');
subplot(3,2,5);mesh(10*log(psg)/255);title('power spectrum 3D');


%% Task 3 - Image Restoration 
%% 1 - add random noise and blur it 
clc; clear; close all;
%Read the image 
im = imread('deer.jpg');
% Convert it to grayscale 
xd = rgb2gray(im);
% show the original image
subplot(2,2,1);imshow(xd);title('original image')
% get the size
[he wi] = size(xd); 

% blur using gaussian smoothing filter 
% gh = imgaussfilt(xd,5);

% other way, by using fspecial function 
kernel = fspecial('Gaussian',[5 5],5); 
% kernel = fspecial('Gaussian',[he wi],5); 
H = kernel;
gh = imfilter(xd,kernel);


% Other way - manual kernel (found online)
% kernel = (1/200).*[1 4 7 4 1; 4 16 26 16 4; 7 26 41 26 7;4 16 26 16 4;1 4 7 4 1]

% Other way - manual kernel (from wikipedia)
% kernel = [0.00000067	0.00002292	0.00019117	0.00038771	0.00019117	0.00002292	0.00000067;
% 0.00002292	0.00078633	0.00655965	0.01330373	0.00655965	0.00078633	0.00002292;
% 0.00019117	0.00655965	0.05472157	0.11098164	0.05472157	0.00655965	0.00019117;
% 0.00038771	0.01330373	0.11098164	0.22508352	0.11098164	0.01330373	0.00038771;
% 0.00019117	0.00655965	0.05472157	0.11098164	0.05472157	0.00655965	0.00019117;
% 0.00002292	0.00078633	0.00655965	0.01330373	0.00655965	0.00078633	0.00002292;
% 0.00000067	0.00002292	0.00019117	0.00038771	0.00019117	0.00002292	0.00000067]


% Create noise that follows a normal distribution
nm = 30; 
n = nm*randn(size(xd));
% r = histogram(n,30); % to verify that it is normally distributed 

% add noise to this blurry image 
imageBlurNoise = double(gh) + n; 

subplot(2,2,2);imshow(gh);title('image with gaussian blur'); 
subplot(2,2,3);imshow(imageBlurNoise/255); title('image gaussian blur + gaussian noise');
subplot(2,2,4); freqz2(kernel);('fourrier transform of the gaussian kernel');

%% 2 frequency transform of the blurring function h (called kernel here) 

% freqtra1 = fft(kernel) % 1d
freqtra2 = fft2(kernel); % 2d

% Should I implement it manually? I am not sure wether  I should or not 
%% 3 calculate the 2D FFT of the degraded image using fft2

fono2 = fft2(imageBlurNoise); % FT 2d
%fono1 = fft(imageBlurNoise); % FT 1d
% center it with fftshift
fonos2 = fftshift(fono2); 
%fonos1 =  fftshift(fono1);

% display the noisy image's power spectrum
ps2 = abs(fonos2).^2;
%ps1 = abs(fonos1).^2; 

figure;subplot(2,2,1);imshow(10*log(ps2)/255);title('power spectrum 2D');
subplot(2,2,2);plot(10*log(ps2)/255); title('power spectrum 1D');
subplot(2,2,3);mesh(10*log(ps2)/255);title('power spectrum 3D');
subplot(2,2,4);imshow(imageBlurNoise/255);title('noisy&blurred image')
%subplot(2,2,4); imshow(fonos/255)
%subplot(2,2,4);imshow(10*log(ps1)/255);title('power spectrum 1D? ');
% WHAT DOES IT REVEAL?

%fftimgbl = fftshift(fft2(imageBlur)); 
%figure; 

%% 4 - Inverse filtering 

fhat = fono2./H;
out = abs(ifft2(fhat)); 
figure;imshow(out/255);


%% Extra - blurring with a function (gaussian)
clc;
[u v] = meshgrid(-1+2/wi:2/wi:1,-1+2/he:2/he:1);
F = fft2(xd);

sigma = 5;
% Blurring Function
H = 1./(2*pi*sigma.^2) .* exp(-(u.^2+v.^2)./(2*sigma.^2)); 
G = F.*H;
% Motion Blurred image 
g = (ifft2(G));
figure;imshow(abs(g))
% Noisy AND Motion Blurred image 
xn = double(imnoise(uint8(abs(g)),'gaussian',0,.002)); 
fn = fft2(xn);

fhat = fn./H;
out = abs(ifft2(fhat)); 
figure;imshow(out/255);
%% Extra - wiener filtering 

xx = double(imageBlurNoise);
r0 = wiener2(abs(xx));
figure;imshow(r0/255);

%% extra ------
% calculate the 2D FFT of the degraded image using fft2
fono2 = fft2(xd);
% center it with fftshift
fonos2 = fftshift(fono2); 
%fonos1 =  fftshift(fono1);

% display the noisy image's power spectrum
ps2 = abs(fonos2).^2;
%ps1 = abs(fonos1).^2; 

figure;subplot(2,2,1);imshow(10*log(ps2)/255);title('power spectrum 2D');
subplot(2,2,2);plot(10*log(ps2)/255); title('power spectrum 1D');
subplot(2,2,3);mesh(10*log(ps2)/255);title('power spectrum 3D');
subplot(2,2,4);imshow(xd);title('original image');

%% Bonus
[m,n] = deconvblind(imageBlurNoise,kernel);
figure; 
subplot(1,2,1); imshow(imageBlurNoise); title('noisy img');
subplot(1,2,2); imshow(m); title('filtered img(dcv)');

powerSpec = log10(abs(fftshift(fft2(m))).^2);

figure;
subplot(1,2,1); imshow(fftshift(abs(ifft2(abs(fft2(m)))))/255); title('Magnitude After Filter');
subplot(1,2,2); mesh(powerSpec);title('Spectrum 3D');
figure; 
imshow(m)





 

