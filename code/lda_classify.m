function [res] = lda_classify(U, w, threshold, data)
    proj_data = U' * data;
    lda_value = w' * proj_data;
    res = (lda_value > threshold);
end
