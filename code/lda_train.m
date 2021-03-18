function [U,S,V,threshold,w,sort_v_data1,sort_v_data2] = lda_train(data1,data2,num_feature)
    n_data1 = size(data1,2);
    n_data2 = size(data2,2);
    [U,S,V] = svd([data1 data2],'econ'); 
    data = S*V';
    U = U(:,1:num_feature); % Add this in
    data1 = data(1:num_feature,1:n_data1);
    data2 = data(1:num_feature,n_data1+1:n_data1+n_data2);
    m_data1 = mean(data1,2);
    m_data2 = mean(data2,2);

    Sw = 0;
    for k=1:n_data1
        Sw = Sw + (data1(:,k)-m_data1)*(data1(:,k)-m_data1)';
    end
    for k=1:n_data2
        Sw = Sw + (data2(:,k)-m_data2)*(data2(:,k)-m_data2)';
    end
    Sb = (m_data1-m_data2)*(m_data1-m_data2)';
    
    [V2,D] = eig(Sb,Sw);
    [lambda,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    v_data1 = w'*data1;
    v_data2 = w'*data2;
    
    if mean(v_data1) > mean(v_data2)
        w = -w;
        v_data1 = -v_data1;
        v_data2 = -v_data2;
    end
    
    % Don't need plotting here
    sort_v_data1 = sort(v_data1);
    sort_v_data2 = sort(v_data2);
    t1 = length(sort_v_data1);
    t2 = 1;
    while sort_v_data1(t1) > sort_v_data2(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold = (sort_v_data1(t1)+sort_v_data2(t2))/2;

    % We don't need to plot results
end

