function [U,S,V,threshold,w,labelset] = lda_multi_train(data,labels,labelset,num_feature)
    % labelset map index to labels
    % values in labelset must be unique
    % inverse_labelset map labels to index in labelset
    inverse_labelset = @(x) find(labelset==x);
    
    n_data = size(data,2);
    [U,S,V] = svd(data,'econ'); 
    data = S*V';
    data = data(1:num_feature, :);
    U = U(:,1:num_feature); % Add this in
    n_group = length(labelset);
    m_all = mean(data, 2); % overall mean
    
    % compute mean for each group
    m_data = zeros(size(data, 1), n_group);
    for i=1:n_group
        m_data(:, i) = mean(data(:, labels==labelset(i)), 2);
    end
    
    % compute SS_within and SS_between
    Sw = 0;
    for i=1:n_data
        Sw = Sw + (data(:,i)-m_data(inverse_labelset(labels(i))))*(data(:,i)-m_data(inverse_labelset(labels(i))))';
    end

    Sb = 0;
    for i=1:n_group
        Sb = Sb + (m_data(i) - m_all)*(m_data(i) - m_all)';
    end
    
    [V2,D] = eig(Sb,Sw);
    [lambda,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    v_data = w'*data;
    
    m_v_data = zeros(n_group, 1);
    for i=1:n_group
        m_v_data(i) = mean(v_data(:, labels==labelset(i)));
    end
    % sort labelset based on mean
    tmp_ = [m_v_data labelset'];
    tmp_ = sortrows(tmp_, 1);
    labelset = tmp_(:, 2)';
    % we don't care mean, but it's in tmp_(:, 1)

    % compute thresholds
    threshold_ = zeros(1, n_group+1);
    threshold_(1) = [-inf];
    for i=1:n_group-1
        % we can do this since labelset is sorted based on mean
        tmp1_ = sort(v_data(labels==labelset(i)));
        tmp2_ = sort(v_data(labels==labelset(i+1)));
        t1 = length(tmp1_);
        t2 = 1;
        while tmp1_(t1) > tmp2_(t2)
            t1 = t1-1;
            t2 = t2+1;
        end
        threshold_(i+1) = (tmp1_(t1)+tmp2_(t2))/2;
    end
    % threshold_ should contain n_group + 1 elements after this line
    threshold_(end) = inf;
    
    % threshold contains the range of values each group can take
    threshold = zeros(n_group, 2);
    for i=1:n_group
        threshold(i, 1) = threshold_(i);
        threshold(i, 2) = threshold_(i+1);
    end
end

