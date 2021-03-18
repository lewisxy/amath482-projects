%% Clean workspace
clear all; close all; clc


%% Load Data and Constants
% load one data at a time
%load('data.nosync/cam_1.mat');
%load('data.nosync/cam_2.mat');
load('data.nosync/cam_3.mat');

video_height = 480;
video_width = 640;

data_raw = data;

%% Plot data
data_label = ['x1(t)'; 'y1(t)'; 'x2(t)'; 'y2(t)'; 'x3(t)'; 'y3(t)'];
figure
for i=1:6
    plot(data_raw(i, :));
    hold on;
end
legend(data_label);
set(gca, 'Fontsize',16);
title('Position of Mass in Videos');
xlabel('t (frame)');
ylabel('position (pixel)');

%% PCA (with SVD)
% de-mean data
m_data = mean(data_raw, 2);
data = data_raw - m_data;

[U,S,V] = svd(data,'econ');
% print singular values
disp(S);

% rank 1 approximation
data_rank1 = S(1,1)*U(:,1)*V(:,1)' + m_data;

% rank 2 approximation
data_rank2 = data_rank1 + S(2,2)*U(:,2)*V(:,2)';

% 2 principle components
n = size(data, 2);
y1 = S(1,1)/sqrt(n-1)*U(:,1);
y2 = S(2,2)/sqrt(n-1)*U(:,2);

% plot apprimations
figure
for i=1:6
    plot(data_rank1(i, :));
    hold on;
end
legend(data_label);
set(gca, 'Fontsize',16);
title('Rank-1 Approximation');
xlabel('t (frame)');
ylabel('position (pixel)');

figure
for i=1:6
    plot(data_rank2(i, :));
    hold on;
end
legend(data_label);
set(gca, 'Fontsize',16);
title('Rank-2 Approximation');
xlabel('t (frame)');
ylabel('position (pixel)');


%% Change of Basis (plot data with first 2 principle components)
data_proj = U'*data;

figure
plot(data_proj(1,:),data_proj(2,:),'k.','MarkerSize',10)
axis equal
hold on
y1_proj = U'*y1;
y2_proj = U'*y2;
c = compass(y1_proj(1),y1_proj(2));
set(c,'Linewidth',4);
c = compass(y2_proj(1),y2_proj(2));
set(c,'Linewidth',4);
set(gca, 'Fontsize',16);
title('Data in First 2 Principle Components');
xlabel('First Principle Component');
ylabel('Second Principle Component');