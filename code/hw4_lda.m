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
digit1 = 5;
digit2 = 6;
features = 20;

digit1_train_data = train_flatten(:, train_labels==digit1);
digit2_train_data = train_flatten(:, train_labels==digit2);

[U,S,V,threshold,w,sort_v_data1,sort_v_data2] = lda_train(digit1_train_data, digit2_train_data, features);

%% test results
digit1_test_data = test_flatten(:, test_labels==digit1);
digit2_test_data = test_flatten(:, test_labels==digit2);
test_input = [digit1_test_data digit2_test_data];

res = lda_classify(U, w, threshold, test_input); % return 1 for digit2, 0 for digit1
truth = [zeros(1, size(digit1_test_data, 2)) ones(1, size(digit2_test_data, 2))];

correct_count = sum(res == truth);
correct_proportion = correct_count / size(test_input, 2);