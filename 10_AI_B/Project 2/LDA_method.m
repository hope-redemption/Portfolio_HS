function w = LDA_method(original_matrix, labels, class,k)
% achieve LDA
    a = labels;
    N = length(class);  % the number of the classes
    [~, col] = size(original_matrix);  % col = image of characteristic = 784
    means_C = zeros(N,col); 
    class_M = cell(N,1); % Put different numbers into different groups
    
    % seperate different value("1","5","8") in different class, and compute mean
    for i = 1:N
        class_M{i,1} = original_matrix(a == class(i),:);
        means_C(i,:) = mean(original_matrix(a == class(i),:));
    end
    
    mean_X = mean(original_matrix); %  Average intensity of each location information point
    % generate empty S_w and S_b
    S_w = zeros(col);
    S_b = zeros(col);
    % compute S_w
    for i = 1:N
        % (class_M{i,1} - mean_X) is Mw -> within-class scatter
        sigema = (class_M{i,1} - mean_X)' * (class_M{i,1} - mean_X);
        S_w = S_w + sigema;
    end
    % compute S_b 
    for i = 1:N
        % (means_C(i,:) - mean_X) is Mb
        % between class scatter
        S_b = S_b + size(class_M{i,1},1) * (means_C(i,:) - mean_X)' * (means_C(i,:) - mean_X);
    end
    
    % computing the LDA projection vector w
    W = pinv(S_w) * S_b; % optimal transformation by solving a generalized eigenvalue problem
    [v,~] = eigs(W); % we need to get the eigenvector
    w = real(v(:,1:k));
end

