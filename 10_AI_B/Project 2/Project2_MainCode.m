%% ASSIGNMENT2 : Classification and Clustering
clear
clc

% load data
load mnist-1-5-8.mat
class = [1,5,8];
image_data = images;
label = labels;

%% PCA

% using PCA_method
PCA_component = PCA_method(image_data', 2); % get first two main components

% cluster the data 
% 1 using k-means
[idx,C] = kmeans(PCA_component,3,'Distance','cityblock');
% % using cityblock
% [idx,C] = kmeans(PCA_component,3);
% generate color
color = lines(6);
figure(1)
hold on
% gscatter: 
% creates a scatter plot of x and y, grouped by g. The inputs x and y are vectors of the same size
% specifies the marker color clr for each group
%gscatter(PCA_component(:,1),PCA_component(:,2),idx,'bgm')
gscatter(PCA_component(:,1),PCA_component(:,2),idx,color(1:3,:))
plot(C(:,1),C(:,2),'k*')
title("PCA + clust")
legend('cluster 1','cluster 2','cluster 3','centroid of each cluster')
hold off

figure(2)
hold on
%gscatter(PCA_component(:,1),PCA_component(:,2),label,'gmb')
gscatter(PCA_component(:,1),PCA_component(:,2),label,color(4:6,:))
plot(C(:,1),C(:,2),'k*')
title("actual data")
legend('number 1','number 5','number 8','centroid of each cluster')
hold off

% 2 using hierarchical clustering
%Compute three clusters of the PCA process data using single-linkage
Z = linkage(PCA_component,'average','euclidean'); %create the linkage tree using average-link
c = cluster(Z,'maxclust',3);
% See how the cluster assignments correspond to the three number: 1 5 8
crosstab(c, label)
% Create a dendrogram plot of Z, and visualize it.
figure(3)
% dendrogram(Z,0)
% cutoff = median([Z(end-2,3) Z(end-1,3)]);
%dendrogram(Z,'ColorThreshold',cutoff)
dendrogram(Z,0,'Orientation','left','ColorThreshold','default')


% 3 using GMM
X = PCA_component;
[n,p] = size(X);
figure(4)
plot(X(:,1),X(:,2),'.','MarkerSize',15);
title('original data');

rng(3);
k = 3; % Number of GMM components
options = statset('MaxIter',1000);

Sigma = {'diagonal','full'}; % Options for covariance matrix type
nSigma = numel(Sigma);

SharedCovariance = {true,false}; 
% Indicator for identical or nonidentical covariance matrices
SCtext = {'true','false'};
nSC = numel(SharedCovariance);

d = 500; % Grid length
x1 = linspace(min(X(:,1))-2, max(X(:,1))+2, d);
x2 = linspace(min(X(:,2))-2, max(X(:,2))+2, d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];

figure(5)
threshold = sqrt(chi2inv(0.99,2));
count = 1;
for i = 1:nSigma
    for j = 1:nSC
        gmfit = fitgmdist(X,k,'CovarianceType',Sigma{i}, ...
            'SharedCovariance',SharedCovariance{j},'Options',options); % Fitted GMM
        clusterX = cluster(gmfit,X); % Cluster index 
        mahalDist = mahal(gmfit,X0); % Distance from each grid point to each GMM component
        % Draw ellipsoids over each GMM component and show clustering result.
        subplot(2,2,count);
        h1 = gscatter(X(:,1),X(:,2),clusterX);
        hold on
            for m = 1:k
                idx = mahalDist(:,m)<=threshold;
                Color = h1(m).Color*0.75 - 0.5*(h1(m).Color - 1);
                h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
                uistack(h2,'bottom');
            end    
        plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
        title(sprintf('Sigma is %s\nSharedCovariance = %s',Sigma{i},SCtext{j}),'FontSize',8)
        legend(h1,{'8','1','5'})
        hold off
        count = count + 1;
    end
end

initialCond1 = [ones(n-8,1); [2; 2; 2; 2]; [3; 3; 3; 3]]; % For the first GMM
initialCond2 = randsample(1:k,n,true); % For the second GMM
initialCond3 = randsample(1:k,n,true); % For the third GMM
initialCond4 = 'plus'; % For the fourth GMM
cluster0 = {initialCond1; initialCond2; initialCond3; initialCond4};

converged = nan(4,1);
figure(6)
for j = 1:4
    gmfit = fitgmdist(X,k,'CovarianceType','full', ...
        'SharedCovariance',false,'Start',cluster0{j}, ...
        'Options',options);
    clusterX = cluster(gmfit,X); % Cluster index 
    mahalDist = mahal(gmfit,X0); % Distance from each grid point to each GMM component
    % Draw ellipsoids over each GMM component and show clustering result.
    subplot(2,2,j);
    h1 = gscatter(X(:,1),X(:,2),clusterX); % Distance from each grid point to each GMM component
    hold on;
    nK = numel(unique(clusterX));
    for m = 1:nK
        idx = mahalDist(:,m)<=threshold;
        Color = h1(m).Color*0.75 + -0.5*(h1(m).Color - 1);
        h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
        uistack(h2,'bottom');
    end
	plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
    legend(h1,{'1','8','5'});
    hold off
    converged(j) = gmfit.Converged; % Indicator for convergence
end

%% LDA
LDA_w = LDA_method(image_data',label,class,2);
% computing the projection score
score = image_data'*LDA_w;
figure(7)
gscatter(score(:,1), score(:,2), label, 'rgb','osd')
legend('Number 1','Number 5','Number 8')
title("LDA - MINST")
figure(8)
[idx,C] = kmeans(score,3);
gscatter(score(:,1),score(:,2),idx,'rgb','osd')
legend('cluster 1','cluster 2','cluster 3')
title("K-means clustering")

% % mvnpdf: Multivariate normal probability density function
% % the class 1: 
% x1 = score(labels==1);
% lm1 = mean(x1); % compute mu
% lstd1 = std(x1); % compute sigma
% class1_pdf = mvnpdf(x1,lm1,lstd1); 
% % the class 5:
% x5 = score(labels==5); 
% lm5 = mean(x5); 
% lstd5 = std(x5); 
% class5_pdf = mvnpdf(x5, lm5, lstd5);
% % the class 8:
% x8 = score(labels==8); 
% lm8 = mean(x8); 
% lstd8 = std(x8); 
% class8_pdf = mvnpdf(x8, lm8, lstd8);
% 
% figure(8); 
% hold on; 
% plot(x1, class1_pdf, 'r.'); 
% plot(x5,class5_pdf,'g.'); 
% plot(x8,class8_pdf,'b.'); 
% hold off

%% two-class classification
new_label = zeros(size(label));
for i = 1:max(size(new_label))
    if(label(i) == 1)
        new_label(i) = 0;
    end
    if(label(i) == 8)
        new_label(i) = 0;
    end
    if(label(i) == 5)
        new_label(i) = 1;
    end    
end

% get 5-fold cross validation setting
k = 5;
% Create indices for the 5-fold cross-validation.
cvIndices = crossvalind('Kfold',new_label,k);
acc_RBF = 0;
acc_linear = 0;
acc_neuralnetwork = 0;
AUC_RBF = 0;      
AUC_linear = 0;
AUC_neuralnetwork = 0;

for i = 1:k
    test_logical = (cvIndices == i); % logical vectors test
    train_logical = ~test_logical; % logical vectors train
    
    original_matrix = image_data'; 
    
    train_image = original_matrix(train_logical,:); % get train images  480x784
    test_image = original_matrix(test_logical,:); % get test images  120x784
    train_label = new_label(train_logical); % get train label
    test_label = new_label(test_logical); % get test label
    
    % fitcsvm supports predictive variable data using kernel function mapping
    % SVM with a RBF kernel
    Model_RBF = fitcsvm(train_image,train_label,'KernelFunction','rbf','KernelScale','auto');
    [label_RBF, scores_RBF] = predict(Model_RBF,test_image);
    [X_RBF,Y_RBF,T_rbf,AUC_rbf] = perfcurve(test_label,scores_RBF(:,2),1);
    AUC_RBF = AUC_RBF + AUC_rbf;
    acc_rbf = sum(label_RBF == test_label)/length(test_label);
    acc_RBF = acc_RBF + acc_rbf;
    figure(8+i)
    hold on
    plot(X_RBF,Y_RBF,'r')

    % SVM with a linear kernel
    model_linear = fitcsvm(train_image,train_label,'KernelFunction','linear','KernelScale','auto');
    [label_linear, scores_linear] = predict(model_linear,test_image);
    [X_linear,Y_linear,T_linear,AUC_Linear] = perfcurve(test_label,scores_linear(:,2),1);
    AUC_linear = AUC_linear + AUC_Linear;
    % acc_Linear = sum(label_linear == test_label)/length(test_label);
    acc_linear = acc_linear + sum(label_linear == test_label)/length(test_label);
    plot(X_linear,Y_linear,'g')

    % neural network classifier with one hidden layer
    net = feedforwardnet(2, 'traingd');
    %net.divideParam.trainRatio = 1; % training set [%]
    %net.divideParam.valRatio = 0; % validation set [%]
    %net.divideParam.testRatio = 0; % test set [%]
    % Configure the net
    net.inputs{1}.processFcns = {}; % modify the process function for inputs
    net.outputs{2}.processFcns = {}; % modify the process function for outputs
    net.layers{1}.transferFcn = 'logsig'; % the transfer function for the first layer
    net.layers{2}.transferFcn = 'softmax'; % the transfer function for the second layer
    net.performFcn = 'crossentropy'; % the loss function
    net.trainParam.lr = 0.1; % learning rate. 
    net.trainParam.epochs = 3000;

    net = train(net, train_image', train_label');
    score_net = net(test_image');
    label_net = (score_net>0.5)';
    [X_neuralnetwork,Y_neuralnetwork,T_net,AUC_net] = perfcurve(test_label,score_net,1);
    AUC_neuralnetwork = AUC_neuralnetwork + AUC_net;
    acc_Neuralnetwork = sum(label_net == test_label)/length(test_label);
    acc_neuralnetwork = acc_neuralnetwork + acc_Neuralnetwork;
    plot(X_neuralnetwork,Y_neuralnetwork,'b')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title("Network/SVM ROC")
    legend('RBF SVM','linear SVM','neuralnetwork')
    hold off
end
% get the result of acc and AUC
acc_RBF = acc_RBF/k
acc_linear = acc_linear/k
acc_neuralnetwork = acc_neuralnetwork/k

AUC_RBF = AUC_RBF/k
AUC_linear = AUC_linear/k
AUC_neuralnetwork = AUC_neuralnetwork/k


%% test other parameter
% choose SVM linear kernel
k = 5;
new_label = zeros(size(label));
for i = 1:max(size(new_label))
    if(label(i) == 1)
        new_label(i) = 0;
    end
    if(label(i) == 8)
        new_label(i) = 0;
    end
    if(label(i) == 5)
        new_label(i) = 1;
    end    
end

cvIndices = crossvalind('Kfold',new_label,k);
test_logical = (cvIndices == 2); 
train_logical = ~test_logical;
original_matrix = image_data';
train_image = original_matrix(train_logical,:);
train_label = new_label(train_logical);
test_image = original_matrix(test_logical,:);
test_label = new_label(test_logical);

model_linear_auto = fitcsvm(train_image,train_label,'KernelFunction','linear', 'KernelScale','auto');
[label_linear_auto, scores_linear_auto] = predict(model_linear_auto,test_image);
[X_linear_auto,Y_linear_auto,T_linear_auto,AUC_linear_auto] = perfcurve(test_label,scores_linear_auto(:,2),1);
acc_linear_auto = sum(label_linear_auto == test_label)/length(test_label);
figure(14)
plot(X_linear_auto,Y_linear_auto,'r')
hold on

model_linear_1 = fitcsvm(train_image,train_label,'KernelFunction','linear','KernelScale',1);
[label_linear_1, scores_linear_1] = predict(model_linear_1,test_image);
[X_linear_1,Y_linear_1,T_linear_1,AUC_linear_1] = perfcurve(test_label,scores_linear_1(:,2),1);
acc_linear_1 = sum(label_linear_1 == test_label)/length(test_label);
plot(X_linear_1,Y_linear_1,'g')

model_linear_5 = fitcsvm(train_image,train_label,'KernelFunction','linear', 'KernelScale',5);
[label_linear_5, scores_linear_5] = predict(model_linear_5,test_image);
[X_linear_5,Y_linear_5,T_linear_5,AUC_linear_5] = perfcurve(test_label,scores_linear_5(:,2),1);
acc_linear_5 = sum(label_linear_5 == test_label)/length(test_label);
plot(X_linear_5,Y_linear_5,'b')

xlabel('False positive rate') 
ylabel('True positive rate')
title("Different KernelScale of SVM ROC")
legend('KernelScale = auto','KernelScale = 1','KernelScale = 5')
hold off
% show the acc and AUC
acc_linear_auto
acc_linear_1
acc_linear_5
AUC_linear_auto
AUC_linear_1
AUC_linear_5