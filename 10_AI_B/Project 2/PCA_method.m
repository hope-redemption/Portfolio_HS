function new_low_dim_matrix = PCA_method(origin_high_dim_matrix,k)
% achieve PCA 
% output information about image with first k leading vectors 
% dim(origin_high_dim_matrix)=600x784
    mean_X = mean(origin_high_dim_matrix); % 600x1
    high_dim_matrix = origin_high_dim_matrix - repmat(mean_X, size(origin_high_dim_matrix,1), 1);
    covariance_X = cov(high_dim_matrix);
    [V, D] = eigs(covariance_X); % number of V = number of image characteristic = 784
    new_low_dim_matrix = high_dim_matrix*V(:,1:k);% get the projection on the first k leading vectors

end

