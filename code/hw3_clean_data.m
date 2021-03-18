%% load data 1

load('data.nosync/cam1_1.mat');
load('data.nosync/cam2_1.mat');
load('data.nosync/cam3_1.mat');

% find the smallest time interval
l = min([size(vidFrames1_1, 4), size(vidFrames2_1, 4), size(vidFrames3_1, 4)]);
% cut the video
vidFrames1_1 = vidFrames1_1(:, :, :, 1:l);
vidFrames2_1 = vidFrames2_1(:, :, :, 1:l);
vidFrames3_1 = vidFrames3_1(:, :, :, 1:l);

%% load data 2

load('data.nosync/cam1_2.mat');
load('data.nosync/cam2_2.mat');
load('data.nosync/cam3_2.mat');

% find the smallest time interval
l = min([size(vidFrames1_2, 4), size(vidFrames2_2, 4), size(vidFrames3_2, 4)]);
% cut the video
vidFrames1_2 = vidFrames1_2(:, :, :, 1:l);
vidFrames2_2 = vidFrames2_2(:, :, :, 1:l);
vidFrames3_2 = vidFrames3_2(:, :, :, 1:l);

%% load data 4

load('data.nosync/cam1_4.mat');
load('data.nosync/cam2_4.mat');
load('data.nosync/cam3_4.mat');

% find the smallest time interval
l = min([size(vidFrames1_4, 4), size(vidFrames2_4, 4), size(vidFrames3_4, 4)]);
% cut the video
vidFrames1_4 = vidFrames1_4(:, :, :, 1:l);
vidFrames2_4 = vidFrames2_4(:, :, :, 1:l);
vidFrames3_4 = vidFrames3_4(:, :, :, 1:l);

%% create gray scale video
video_height = 480;
video_width = 640;
% l = 226;
vid_gray = zeros([video_height, video_width, l], 'uint8');
for i=1:l
    vid_gray(:, :, i) = rgb2gray(vidFrames2_4(:, :, :, i));
end
%implay(vid_gray)

%%
% choose a frame where bucket is not blur
% sg = vid_gray(:, :, 10); % cam1_1
% sg = vid_gray(:, :, 108); % cam1_1 part 2 start frame 100
% sg = vid_gray(:, :, 19); % cam2_1
% sg = vid_gray(:, :, 159); % cam2_1 part 2 start frame 150
% sg = vid_gray(:, :, 9); % cam3_1
% sg = vid_gray(:, :, 110); % cam3_1 part 2 start frame 100

% sg = vid_gray(:, :, 11); % cam1_2
% sg = vid_gray(:, :, 37); % cam2_2
% sg = vid_gray(:, :, 18); % cam3_2

% sg = vid_gray(:, :, 53); % cam1_4
% sg = vid_gray(:, :, 72); % cam1_4 part 2
% sg = vid_gray(:, :, 254); % cam1_4 part 3
% sg = vid_gray(:, :, 11); % cam1_4 part 4

sg = vid_gray(:, :, 77); % cam2_4
imshow(sg); axis on;

%% find bucket (manually)
% bucket = sg(220:300, 320:380); % cam1_1
% bucket = sg(350:425, 312:372); % cam1_1 part 2
% bucket = sg(110:195, 260:330); % cam2_1 part 1
% bucket = sg(270:360, 270:335); % cam2_1 part 2
% bucket = sg(245:310, 275:350); % cam3_1 part 1
% bucket = sg(265:328, 405:478); % cam3_1 part 2

% bucket = sg(310:390, 317:370); % cam1_2
% bucket = sg(305:407, 237:318); % cam2_2
% bucket = sg(240:295, 407:478); % cam3_2

% bucket = sg(245:312, 332:380); % cam1_4
% bucket = sg(328:402, 350:405); % cam1_4 part 2
% bucket = sg(252:330, 382:438); % cam1_4 part 4

bucket = sg(242:340, 272:352); % cam1_4 part 4
imshow(bucket);
% save('data.nosync/something_else', 'bucket');

%% compute conv match
numFrames = l;
idxs = zeros(1, numFrames);
for j = 1:numFrames
    frame = vid_gray(:,:,j);
    tmp = convMatch(frame, bucket);
    [m, id] = max(tmp(:));
    idxs(j) = id;
    disp(j);
end
% save('data.nosync/somethingelse', 'idxs');

%% plot image
figure
[tmp1, tmp2] = ind2sub([video_height video_width], idxs);
plot(tmp1, 'red')
hold on;
plot(tmp2, 'blue')

%% interpolate data (fixing data)
t = 1:l;
% remove outliers in both t and data (for non-uniform interpolation)
data = [1:l; orig_data];
% condition is customized to data
data = data(:, orig_data > 300 & orig_data < 450); % cam1_1_x
% data = data(:, orig_data > 250); % cam1_1_y
out = resample(data(2, :), data(1, :), 1, 1, 1); % interpolate data
out = round(out); % make them integer

%% plot things out
numFrames = l;
for j = 1:numFrames
    frame = vid_gray(:,:,j);
%     tmp = (frame(:) >= uint8(round(255 * 0.98)));
%     idx = find(tmp);
%     [iy, ix] = ind2sub([video_height video_width], idx);
    imshow(frame);
    hold on;
%     scatter(ix, iy, 1, 'red'); % raw pixels
    scatter(tmp2(j), tmp1(j), 10, 'blue', 'filled'); % conv result
    hold off;
    drawnow
    pause(1 / 30);
end