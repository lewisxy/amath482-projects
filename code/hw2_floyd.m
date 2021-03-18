%% Cleanup workspace
clear all; close all; clc

%% Step 1: load and prepare data
[y, Fs] = audioread('data.nosync/Floyd.m4a');
resample_factor = 1;

%% Option 1: truncate the data
% the entire audio sample is large, we truncate it to
% avoid memory error
y = y(1:(Fs * 15)); % sample first 15 seconds

%% Option 2: downsample the data
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

% convert frequency to piano key number
% https://en.wikipedia.org/wiki/Piano_key_frequencies
freq2music = @(x) (12 .* log2(x / 440) + 49);
music2freq = @(x) (2.^((x - 49) / 12) .* 440);
log_ks = freq2music(abs(ks));

%% Step 2: create a spectrogram
% because of Fourier uncertainity, the less uncertainty in time means 
% more uncertainty in frequency, we adjust parameter a to get a good
% balance of both

% to make the result consistent, we adjust a based on resample_factor
%a = 5120;
%a = 160; % high freq resolution
a = 2560;
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
set(gca,'ylim', [max(ks(1), -1200) min(ks(end), 1200)], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title('Spectrogram for Floyd');

%% Step 3: adjust spectrogram to figure out musical note
% plot spectrogram with frequency axis rescaled to piano key number
figure()
pcolor(tau, log_ks, sgt_spec) % ks
shading interp
set(gca,'ylim', [1 88], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('piano key number')
title('Spectrogram for Floyd With Piano Key Number');

% plot lines to indicate piano key
hold on;
for i=1:88
    line([0 L], [i i], 'Color', 'white')
    if mod(i, 12) == 1 % highlight every octave
        line([0 L], [i i], 'Color', 'green')
    end
end
line([0 L], [49 49], 'Color', 'red') % highlight key 49 at 440 Hz

%% plot again with zoomed in view (bass)
lb = 12; ub = 42; % roughly 60 to 250 Hz

figure()
pcolor(tau, log_ks, sgt_spec) % ks
shading interp
set(gca,'ylim', [lb ub], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('piano key number')
title('Spectrogram for Floyd with Piano Key Number (Zoomed)');

% plot lines to indicate piano key
hold on;
for i=lb:ub
    line([0 L], [i i], 'Color', 'white')
    if mod(i, 12) == 1 % highlight every octave
        line([0 L], [i i], 'Color', 'green')
    end
end

% Music note being played are:
% 27 25 23 22 20 34 35 39
% after removing overtones
% 27 25 23 22 20

%% filter bass
% bass typically has a range of 60 to 250 Hz
box_filter = abs(ks) <= 250 & abs(ks) >= 60;
% apply filter
st = fftshift(fft(s));
sft = st.*box_filter;
sf = ifft(ifftshift(sft));

% sf contains the audio for isolated bass
% we can play it (by uncommenting the next line)
%p8 = audioplayer(sf, Fs); playblocking(p8);

%% plot again with zoomed in view (guitar)
lb = 40; ub = 64;

figure()
pcolor(tau, log_ks, sgt_spec) % ks
shading interp
set(gca,'ylim', [lb ub], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('piano key number')
title('Spectrogram for Floyd with Piano Key Number (Zoomed)');

% plot lines to indicate piano key
hold on;
for i=lb:ub
    line([0 L], [i i], 'Color', 'white')
    if mod(i, 12) == 1 % highlight every octave
        line([0 L], [i i], 'Color', 'green')
    end
end
line([0 L], [49 49], 'Color', 'red')

% Music note being played are:
% 27 25 23 22 20 34 35 39

%% removing overtones (experimental)
% try to subtract the bass frequency region from the guitar region

% the range of notes we want to remove
bass_notes = 20:27;

out = zeros(size(sgt_spec));
for j = 1:length(tau)
    % copy spectrogram
    out(:, j) = sgt_spec(:, j);
    for i=1:length(bass_notes)
        % compute note interval
        note_lb = bass_notes(i) - 0.5;
        note_ub = bass_notes(i) + 0.5;
        % compute corresponding frequencies
        mask_note = log_ks' > note_lb & log_ks' < note_ub;
        res = mask_note .* sgt_spec(:, j);
        count_note = sum(mask_note);
        notes = nonzeros(res);
        % remove up to 3 overtones (2nd, 4th, 8th)
        for k=1:3
            % compute the destination note and frequency range
            overtone_note_lb = note_lb + 12 * k;
            overtone_note_ub = note_ub + 12 * k;
            mask_overtone = log_ks' > overtone_note_lb & log_ks' < overtone_note_ub;
            res = mask_overtone .* sgt_spec(:, j);
            count_overtone = sum(mask_overtone);
            % interpolate values so that it fit the new range
            overtone_notes = resample(notes, count_overtone, count_note);
            % apply subtraction, clip to 0
            out(mask_overtone, j) = max(0, out(mask_overtone, j) - overtone_notes);
        end
    end
end

%% plot again with zoomed in view (guitar) again after removing overtones
lb = 40; ub = 64;

figure()
pcolor(tau, log_ks, out) % ks
shading interp
set(gca,'ylim', [lb ub], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('piano key number')
title('Spectrogram for Floyd with Piano Key Number (Zoomed)');

% plot lines to indicate piano key
hold on;
for i=lb:ub
    line([0 L], [i i], 'Color', 'white')
    if mod(i, 12) == 1 % highlight every octave
        line([0 L], [i i], 'Color', 'green')
    end
end
line([0 L], [49 49], 'Color', 'red')

% Music note being played are:
% 42 44 46 47 54