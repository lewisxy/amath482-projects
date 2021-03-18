%% setup
clear; close all; clc;

%% load data
[train_images, train_labels] = mnist_parse('data.nosync/train-images.idx3-ubyte', 'data.nosync/train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('data.nosync/t10k-images.idx3-ubyte', 'data.nosync/t10k-labels.idx1-ubyte');

%% flatten data: turn image to vectors
train_flatten = zeros(size(train_images, 1) * size(train_images, 2), size(train_images, 3));
for i=1:size(train_images, 3)
    tmp_ = train_images(:, :, i);
    train_flatten(:, i) = tmp_(:);
end

test_flatten = zeros(size(test_images, 1) * size(test_images, 2), size(test_images, 3));
for i=1:size(test_images, 3)
    tmp_ = test_images(:, :, i);
    test_flatten(:, i) = tmp_(:);
end

%% train LDA
%digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; % 0.2550
% digits = [1, 3, 5]; % 0.7511
% digits = [1, 4, 9]; % 0.7105
n_group = length(digits);
features = 20;

train_data_size = zeros(1, n_group);

for i=1:n_group
    tmp_ = train_flatten(:, train_labels==digits(i));
    train_data_size(i) = size(tmp_, 2);
end

% allocate space for data
train_data_extracted = zeros(size(train_flatten, 1), sum(train_data_size));
tmp_ = 1;
% build training data (and labels)
for i=1:n_group
    train_data_extracted(:, tmp_:tmp_+train_data_size(i)-1) = train_flatten(:, train_labels==digits(i));
    train_labels_extracted(tmp_:tmp_+train_data_size(i)-1) = digits(i);
    tmp_ = tmp_ + train_data_size(i);
end

[U,S,V,threshold,w,labelset] = lda_multi_train(train_data_extracted, train_labels_extracted, digits, features);

%% test results
test_data_size = zeros(1, n_group);

for i=1:n_group
    tmp_ = test_flatten(:, test_labels==digits(i));
    test_data_size(i) = size(tmp_, 2);
end

% allocate space for data
test_data_extracted = zeros(size(test_flatten, 1), sum(test_data_size));
tmp_ = 1;
% build testing data (and labels)
for i=1:n_group
    test_data_extracted(:, tmp_:tmp_+test_data_size(i)-1) = test_flatten(:, test_labels==digits(i));
    test_labels_extracted(tmp_:tmp_+test_data_size(i)-1) = digits(i);
    tmp_ = tmp_ + test_data_size(i);
end

res = lda_multi_classify(U, w, threshold, labelset, test_data_extracted);

correct_count = sum(res == test_labels_extracted);
correct_proportion = correct_count / size(test_data_extracted, 2);