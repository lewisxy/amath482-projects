function [res] = lda_multi_classify(U, w, threshold, labelset, data)
    % assume labelset is sorted
    proj_data = U' * data;
    size(w')
    size(proj_data)
    lda_value = w' * proj_data;
    res = zeros(1, size(data, 2));
    for i=1:length(labelset)
        lb = threshold(i, 1);
        ub = threshold(i, 2);
        tmp_ = (lda_value > lb & lda_value < ub);
        res(tmp_) = labelset(i);
    end
end
