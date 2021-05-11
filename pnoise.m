% This function is used to create a matrix of periodical noise
% It needs to be added to the  original image.
% the parameters are the size of the matrix to be created (same size as the image),
% the coefficient and the frequency
function n = pnoise(f0, n1, n2,A)
% close all;clear all;clc

[t1, t2] = meshgrid(1:n2,1:n1);
n = A*sin(2*pi*f0*((t1-1)/n2 + (t2-1)/n1));

%figure;imagesc(n)

fn = fftshift(fft2(n));

% figure;imagesc(abs(fn));
% figure;mesh(abs(fn))
% figure;plot(abs(fn));
