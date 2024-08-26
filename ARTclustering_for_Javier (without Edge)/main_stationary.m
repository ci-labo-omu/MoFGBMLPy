%
% ART-based Clustering for Javier (without Edge)
% 
% No reference for this algorithm.
% I made some modifications to a clustering algorithm proposed in:
% https://doi.org/10.48550/arXiv.2305.01507
% 


% clc
clear
close all
whitebg('white')



TRIAL = 1;    % Number of trials

NR = 0.0; % Noise Rate [0-1]


load 2D_ClusteringDATASET; numD = 5000; 
% load 2D_ClusteringDATASET; numD = 15000;
data = [data(1:end,1) data(1:end,2)];
originDATA = data;

data = [data(1:end,1) data(1:end,2)];

d1 = data(1:numD,:);
d2 = data(15001:15000+numD,:);
d3 = data(30001:30000+numD,:);
d4 = data(45001:45000+numD,:);
d5 = data(60001:60000+numD,:);
d6 = data(75001:75000+numD,:);
D = [d1; d2; d3; d4; d5; d6];
data = D;


% Normalization [0-1]
data = normalize(data,'range');


ran = randperm(size(data,1));
data = data(ran,:);
data = data(1:size(data,1),:);

% Noise Setting [0,1]
if NR > 0
    noiseDATA = rand(size(data,1)*NR, size(data,2));
    data(1:size(noiseDATA,1),:) = noiseDATA;
end



% Parameters ========================================================
net.numNodes    = 0;   % the number of nodes
net.weight      = [];  % node position
net.CountNode = [];    % winner counter for each node
net.adaptiveSig = [];  % kernel bandwidth for CIM in each node
net.LabelCluster = [];

net.Lambda = 50;       % an interval for calculating a kernel bandwidth for CIM
net.minCIM = 0.15;      % similarity threshold
% ====================================================================


time_train = 0;


for traial = 1:TRIAL
    
    fprintf('Iterations: %d/%d\n',traial,TRIAL);

    % Randamize data
    rng(11)
    ran = randperm(size(data,1));
    data = data(ran,:);
      
    % Training ==========================================
    tic
    net = ARTclustering_woEdge_Train(data, net);
    time_train = time_train + toc;
    % ===================================================
    
    % Results
    resultNumNodes = ['   Num. Clusters:', num2str(net.numNodes)];
    disp(resultNumNodes);
    disp([' Processing Time: ', num2str(time_train)]);
    disp('');
    
end


% Create a figure with specified size
figure(1);
set(gcf, 'Position', [250, 200, 1200, 600]); % [left, bottom, width, height]
set(gcf, 'Color', [1 1 1]);
cla;

% Plot network
subplot(1, 2, 1);  % 1 row, 2 columns, first plot
myPlot_withoutEdge(data, net);

% Plot estimated density
subplot(1, 2, 2);  % 1 row, 2 columns, second plot
estimateDensityByCountNode(net)
axis equal







