%% setup
clear; close all; clc;

%% load data
[train_images, train_labels] = mnist_parse('data.nosync/train-images.idx3-ubyte', 'data.nosync/train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('data.nosync/t10k-images.idx3-ubyte', 'data.nosync/t10k-labels.idx1-ubyte');

%% flatten data: turn image to vectors
test_flatten = zeros(size(test_images, 1) * size(test_images, 2), size(test_images, 3));
for i=1:size(test_images, 3)
    tmp_ = test_images(:, :, i);
    test_flatten(:, i) = tmp_(:);
end

train_flatten = zeros(size(train_images, 1) * size(train_images, 2), size(train_images, 3));
for i=1:size(train_images, 3)
    tmp_ = train_images(:, :, i);
    train_flatten(:, i) = tmp_(:);
end

%% remove mean of the data
test_mean = mean(test_flatten, 2);
test_demean = test_flatten - test_mean;
train_mean = mean(train_flatten, 2);
train_demean = train_flatten - train_mean;

%% perform SVD
% [U,S,V] = svd(test_demean,'econ');
[U,S,V] = svd(train_demean,'econ');
% U (784, 784) is eigenvectors
% S (784, 784) is diagonal matrix with eigenvalues
% V (10000, 784) is how data represented in eigen space (spaned by U)
% we can use V as training data for classification (LDA, ...)
% as SVD makes it is quite separatable

%% visualize 10 largest eigenvectors
for i=1:10
    tmp_ = U(:, i);
    tmp_ = reshape(tmp_, 28, 28);
    imshow(tmp_, []); % show with adaptative range
end

%% plot distribution of singular values
tmp_ = diag(S);
subplot(2, 1, 1);
plot(tmp_)
title('Singular values for training data (after subtracting mean)')
ylabel('singular value')
xlabel('index of singular of value')
subplot(2, 1, 2);
semilogy(tmp_)
title('Singular values for training data in log scale (after subtracting mean)')
ylabel('singular value (in log scale)')
xlabel('index of singular of value')

%% test reconstruction
% reconstruct image 1 from SVD result
tmp_ = U * S * V(1, :)';
imshow(reshape(tmp_, 28, 28), [])
% this is the same as
imshow(reshape(test_demean(:, 1), 28, 28), [])

% reconstruct with k modes
k = 100;
tmp_ = U(:, 1:k) * S(1:k, 1:k) * V(1, 1:k)';
imshow(reshape(tmp_, 28, 28), [])

% visualize difference
plot(tmp_)
hold on;
plot(test_demean(:, 1))

% reconstruction error (MSE)
mean((tmp_ - test_demean(:, 1)).^2)

%% visualize first 10 PCA modes
figure
suptitle('First 10 PCA modes');
for i=1:10
    subplot(2, 5, i);
    imshow(reshape(U(:, i), 28, 28), [])
    title(num2str(i));
end

%% visualize projection onto 3D
% set seed for color
% rng(123456);
rng(4);
colors = rand(10, 3);
% v_idx = [1, 3, 5];
v_idx = [2, 3, 5];
% v_idx = [2, 2, 2];

v_modes = V(:, v_idx);
% scatter3(v_modes(:, 1), v_modes(:, 2), v_modes(:, 3), 1, colors(test_labels + 1))

% plot 10 times to make it easier for legend
figure
for i=(1:10)
    scatter3(v_modes(train_labels==i-1, 1), v_modes(train_labels==i-1, 2), v_modes(train_labels==i-1, 3), 3, colors(i, :))
    if i == 1
        hold on;
    end
end
hold off;

tmp_ = num2str(((1:10)-1)');
% tmp_ = mat2cell(tmp_, ones(1,10), 1);
tmp_ = num2cell(tmp_);
legend(tmp_{:});
% to make it easier to view the resulted plot
rotate3d on;
title('visualize projection of data onto 2,3,5th V-modes')
xlabel('2nd eigenvector');
ylabel('3rd eigenvector');
zlabel('5th eigenvector');

%% LDA
digit1 = 0;
digit2 = 1;

% Project onto PCA modes
feature = 20;

proj_data = S*V'; % projection onto principal components: X = USV' --> U'X = SV'

digit1_data = proj_data(1:feature, test_labels==digit1);
digit2_data = proj_data(1:feature, test_labels==digit2);

% number of data vector for each catagories
n_digit1 = size(digit1_data, 2);
n_digit2 = size(digit2_data, 2);

% Calculate scatter matrices
m_digit1 = mean(digit1_data, 2);
m_digit2 = mean(digit2_data, 2);

Sw = 0; % within class variances
for k = 1:n_digit1
    Sw = Sw + (digit1_data(:,k) - m_digit1)*(digit1_data(:,k) - m_digit1)';
end
for k = 1:n_digit2
    Sw = Sw + (digit2_data(:,k) - m_digit2)*(digit2_data(:,k) - m_digit2)';
end

Sb = (m_digit1-m_digit2)*(m_digit2-m_digit2)'; % between class

%% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

v_digit1_data = w'*digit1_data;
v_digit2_data = w'*digit2_data;

%% Make digit1 below the threshold

if mean(v_digit1_data) > mean(v_digit2_data)
    w = -w;
    v_digit1_data = -v_digit1_data;
    v_digit2_data = -v_digit2_data;
end

%% Plot projections (not for function)

figure
plot(v_digit1_data,zeros(n_digit1),'ob','Linewidth',2)
hold on
plot(v_digit2_data,ones(n_digit2),'dr','Linewidth',2)
ylim([0 1.2])

%% Find the threshold value

sort_digit1 = sort(v_digit1_data);
sort_digit2 = sort(v_digit2_data);

t1 = length(v_digit1_data);
t2 = 1;
while sort_digit1(t1) > sort_digit2(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort_digit1(t1) + sort_digit2(t2))/2;

%% Plot histogram of results

figure
subplot(1,2,1)
histogram(sort_digit1,30); hold on, plot([threshold threshold], [0 210],'r')
% set(gca,'Xlim',[-300 400],'Ylim',[0 300],'Fontsize',14)
title('digit1')
subplot(1,2,2)
histogram(sort_digit2,30); hold on, plot([threshold threshold], [0 210],'r')
% set(gca,'Xlim',[-300 400],'Ylim',[0 300],'Fontsize',14)
title('digit2')

%%
