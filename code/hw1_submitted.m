%% Clean workspace
clear all; close all; clc

% Imports the data as the 262144x49 (space by time) matrix called subdata 
load('data.nosync/subdata.mat');

%% Setup constants
L = 10; % spatial domain
n = 64; % Fourier modes
ntime = 49; % number of time instance
x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
[X,Y,Z] = meshgrid(x,y,z); % grid for space
[Kx,Ky,Kz] = meshgrid(ks,ks,ks); % grid for frequency

%% Step 1: locate the frequency signature by averaging in frequency space
ut_sum = complex(zeros(n, n, n));
for j = 1:ntime
    uj(:,:,:) = reshape(subdata(:,j), n, n, n);
    utj = fftn(uj);
    ut_sum = ut_sum + utj;
end
ut_avg = ut_sum / ntime;
utm_avg = abs(ut_avg); % signal magnitude in frequency space

tmp_ = fftshift(utm_avg);
[mv, idx] = max(tmp_(:));
[i1, i2, i3] = ind2sub([n, n, n], idx);
% Note: meshgrid treat horizontal axis as x and vertical axis as y
% this is different from index notation (row, column)
% to keep consistency with meshgrid, we swap x and y from index
fx = ks(i2); fy = ks(i1); fz = ks(i3); % actual center frequency
fprintf('frequency signature: (%.4f, %.4f, %.4f)\n', fx, fy, fz);

%% Step 2: make a gaussian filter using the frequency signature we found
% the larger tau the faster filter decays
tau = 0.1;
F = exp(-tau*((Kx - fx).^2 + (Ky - fy).^2 + (Kz - fz).^2));

%% Step 3: track the submarine by applying filter in each timer instance
mvs = zeros(ntime, 1);
idxs = zeros(ntime, 1);
for j = 1:ntime
    uj(:,:,:) = reshape(subdata(:,j), n, n, n);
    utj = fftshift(fftn(uj));
    utfj = F.*utj; % apply filter
    ufj = ifftn(ifftshift(utfj));
    tmp_ = abs(ufj);
    [mv, idx] = max(tmp_(:));
    mvs(j) = mv;
    idxs(j) = idx; % record the position at each measurement
end

% compute the path
% note: we swap x and y in index notation to be consistent with meshgrid
[iy, ix, iz] = ind2sub([n, n, n], idxs);

%% plot the path of submarine 3D
figure;
plot3(x(ix), y(iy), z(iz), '-o');
hold on;
plot3(x(ix(end)), y(iy(end)), z(iz(end)), 'o', 'Color', 'red');
axis([x(1) x(end) y(1) y(end) z(1) z(end)]);
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
legend('path', 'last position', 'Location','northwest');
title('Path of Submarine (3D)');

%% plot the path of submarine 2D
figure;
plot(x(ix), y(iy), '-o');
hold on;
plot(x(ix(end)), y(iy(end)), 'o', 'Color', 'red');
axis([x(1) x(end) y(1) y(end)]);
xlabel('x');
ylabel('y');
legend('path', 'last position', 'Location','northwest');
title('Path of Submarine (2D)');

%% print positions
for j = 1:ntime
    fprintf('%d: position: (%.4f, %.4f, %.4f)\n', j, x(ix(j)), y(iy(j)), z(iz(j)));
end