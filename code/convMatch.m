function [ dst_img ] = convMatch(src_img, pattern_img)
%CONVMATCH Summary of this function goes here
%   src_img is expected to be larger than or equal to pattern_img in both
%   dimension
    src_img = double(src_img) / 255;
    pattern_img = double(pattern_img) / 255;
    offset_i = round(size(pattern_img, 1) / 2);
    offset_j = round(size(pattern_img, 2) / 2);
    pattern_normalized = pattern_img / norm(pattern_img(:));
    dst_img = zeros(size(src_img));
    for i=1:(size(src_img, 1) - size(pattern_img, 1))
        for j=1:(size(src_img, 2) - size(pattern_img, 2))
            tmp1 = src_img(i:i+size(pattern_img,1)-1, j:j+size(pattern_img,2)-1);
            tmp1 = tmp1 / norm(tmp1(:));
            dst_img(i+offset_i, j+offset_j) = sum(tmp1(:) .* pattern_normalized(:));
        end
    end
end

