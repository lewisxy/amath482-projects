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

%% train SVM
% digit1 = 4; % 3
% digit2 = 9; % 5
digit1 = 3; % 3
digit2 = 5; % 5
features = 784;

digit1_train_data = train_flatten(1:features, train_labels==digit1);
digit2_train_data = train_flatten(1:features, train_labels==digit2);

train_input = [digit1_train_data digit2_train_data]';
train_truth = [zeros(1, size(digit1_train_data, 2)) ones(1, size(digit2_train_data, 2))]';

SVMModel = fitcsvm(train_input ,train_truth);

%% test results
digit1_test_data = test_flatten(1:features, test_labels==digit1);
digit2_test_data = test_flatten(1:features, test_labels==digit2);

test_input = [digit1_test_data digit2_test_data]';

[~, score] = predict(SVMModel, test_input);
truth = [zeros(1, size(digit1_test_data, 2)) ones(1, size(digit2_test_data, 2))]';

correct_count = sum((score(:, 2) > 0) == (truth == 1)); % 1751 outof 1991
correct_proportion = correct_count / size(test_input, 1); % 0.8795

% result: 4,9: 0.8795, 3,5: 0.8386