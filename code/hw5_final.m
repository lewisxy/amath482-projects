% Warning: this program consumes a large amount (> 10GB) of memory
%% setup
clear; close all; clc;

%% read video to matrix
%v = VideoReader('data.nosync/ski_drop_low_low.mp4');
v = VideoReader('data.nosync/monte_carlo_low_low.mp4');
n_frame = round(v.Duration * v.FrameRate);
data = zeros(v.Width * v.Height, n_frame);
for i=1:n_frame
    tmp_ = readFrame(v);
    tmp_ = double(rgb2gray(tmp_)) / 255;
    data(:, i) = tmp_(:);
end

%% preparing data
dt = 1 / v.FrameRate;
t = linspace(0, v.Duration, n_frame);
X1 = data(:,1:end-1);
X2 = data(:,2:end);

%% performing SVD
[U, Sigma, V] = svd(X1,'econ');
S = U'*X2*V*diag(1./diag(Sigma));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U*eV;

%% create DMD solution (for all)
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions

u_modes = zeros(length(y0),length(t));
for iter = 1:length(t)
   u_modes(:,iter) = y0.*exp(omega*t(iter)); % y0 .* mu .* exp(iter)
end
u_dmd = Phi*u_modes;

%% reconstruct background
% fixed cutoff value (depends on both length and resolution of video)
%idx_bg = abs(omega) < 10;
% first 3 smallest values (depends only on length of video)
tmp_ = sort(abs(omega));
idx_bg = abs(omega) <= tmp_(3);

y0_bg = Phi(:, idx_bg)\X1(:,1); % pseudoinverse to get initial conditions

u_modes_bg = zeros(length(y0_bg),length(t));
for iter = 1:length(t)
   u_modes_bg(:,iter) = y0_bg.*exp(omega(idx_bg)*t(iter));
end
u_dmd_bg = Phi(:, idx_bg)*u_modes_bg;

%% get real values for foreground and background
% below are different approaches, uncomment them to sum
% make both foreground (X_sparse) and background (X_low_rank) nonnegative
% approach 1: absolute value of difference
u_dmd_fg = abs(u_dmd - u_dmd_bg);
%u_dmd_fg = abs(data - u_dmd_bg); % using original video also works
u_dmd_bg = abs(u_dmd_bg);

% approach 2: difference of absolute value (with normalization)
% u_dmd_bg = abs(u_dmd_bg);
% u_dmd_fg = abs(u_dmd) - u_dmd_bg;
%u_dmd_fg = data - u_dmd_bg; % using original video also works

% linear standardization (this preserve the scale of differences)
% (x - min(x)) / (max(x) - min(x)) for each feature
% imshow automatically does this if using imshow(img, [])
u_dmd_fg = (u_dmd_fg - min(u_dmd_fg, [], 1)) ./ ...
    (max(u_dmd_fg, [], 1) - min(u_dmd_fg, [], 1));

% approach 3: residual (I don't think this works)
% u_dmd_fg = abs(u_dmd) - abs(u_dmd_bg);
% r = (u_dmd_fg < 0).*u_dmd_fg;
% u_dmd_fg(u_dmd_fg < 0) = 0;
% u_dmd_bg = abs(u_dmd_bg) + r;

%% visualize background
figure
imshow(reshape(u_dmd_bg(:, 1), v.Height, v.Width), [])
title('background extracted');

% play video
%implay(reshape(u_dmd_bg, v.Height, v.Width, n_frame))

%% visualize foreground
figure
imshow(reshape(u_dmd_fg(:, 1), v.Height, v.Width), [])
% disable adaptative
%imshow(reshape(u_dmd_fg(:, 1), v.Height, v.Width))
title('foreground extracted');

% play video
%implay(reshape(u_dmd_fg, v.Height, v.Width, n_frame))

%% visualize original frame
figure
imshow(reshape(data(:, 1), v.Height, v.Width), [])
title('original video');

% play video
%implay(reshape(data, v.Height, v.Width, n_frame))