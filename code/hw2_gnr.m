%% Cleanup workspace
clear all; close all; clc

%% Step 1: load and prepare data
[y, Fs] = audioread('data.nosync/GNR.m4a');
resample_factor = 1;

%% Option 1: downsample the data
% downsample data to make it easier to work with
% MATLAB will take a lot of time to plot spectrogram with original data
% and likely to produce memory error
resample_factor = 1/4; % 1 means original, < 1 means downsample
y = resample(y, resample_factor * Fs, Fs);
Fs = resample_factor * Fs;

%% Step 1 (continue): load constants
L = length(y) / Fs; % length of the song in sec
n = length(y); %  total number of samples
ts = (1:length(y)) / Fs; % time domain
if mod(n, 2) == 0
    k = (1/L)*[0:n/2-1 -n/2:-1]; % even length
else
    k = (1/L)*[0:n/2 -n/2:-1]; % odd length
end
ks = fftshift(k); % frequency domain

%% Step 2: create a spectrogram
% because of Fourier uncertainity, the less uncertainty in time means 
% more uncertainty in frequency, we adjust parameter a to get a good
% balance of both

a = 10240;
tau = 0:0.1:ts(end);

s = y';
sgt_spec = zeros(length(ts), length(tau));
for j = 1:length(tau)
   g = exp(-a*(ts - tau(j)).^2); % Window function
   sg = g.*s;
   sgt = fft(sg);
   sgt_spec(:, j) = fftshift(abs(sgt));
end

%% Step 2 (continue): plot spectrogram
figure()
pcolor(tau, ks, sgt_spec)
shading interp
set(gca,'ylim', [ks(1) ks(end)], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title('Spectrogram for GNR');

%% Step 3: adjust spectrogram to figure out musical note
% convert frequency to piano key number
% https://en.wikipedia.org/wiki/Piano_key_frequencies
freq2music = @(x) (12 .* log2(x / 440) + 49);
log_ks = freq2music(abs(ks));

% plot spectrogram with frequency axis rescaled to piano key number
figure()
pcolor(tau, log_ks, sgt_spec) % ks
shading interp
set(gca,'ylim', [1 88], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('piano key number')
title('Spectrogram for GNR With Piano Key Number');

% plot lines to indicate piano key
hold on;
for i=1:88
    line([0 L], [i i], 'Color', 'white')
    if mod(i, 12) == 1 % highlight every octave
        line([0 L], [i i], 'Color', 'green')
    end
end
line([0 L], [49 49], 'Color', 'red') % highlight key 49 at 440 Hz

%% plot again with zoomed in view
lb = 28; ub = 64;

figure()
pcolor(tau, log_ks, sgt_spec) % ks
shading interp
set(gca,'ylim', [lb ub], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('piano key number')
title('Spectrogram for GNR with Piano Key Number (Zoomed)');

% plot lines to indicate piano key
hold on;
for i=lb:ub
    line([0 L], [i i], 'Color', 'white')
    if mod(i, 12) == 1 % highlight every octave
        line([0 L], [i i], 'Color', 'green')
    end
end
line([0 L], [49 49], 'Color', 'red') % highlight key 49 at 440 Hz

% Music note being played are:
% 41 43 46 48 53 57 58 59