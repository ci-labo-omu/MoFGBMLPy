%
% ART-based Clustering for Javier (with Edge)
% 
% No reference for this algorithm.
% I made some modifications to a clustering algorithm proposed in:
% https://doi.org/10.48550/arXiv.2305.01507
% 


% clc
clear
% close all
whitebg('white')



TRIAL = 1;    % Number of trials

NR = 0.1; % Noise Rate [0-1]


load 2D_ClusteringDATASET; numD = 5000; c=6;
% load 2D_ClusteringDATASET; numD = 15000; c=6;
DATA = [data(1:end,1) data(1:end,2)];
originDATA = DATA;

DATA = [data(1:end,1) data(1:end,2)];

d1 = DATA(1:numD,:);
d2 = DATA(15001:15000+numD,:);
d3 = DATA(30001:30000+numD,:);
d4 = DATA(45001:45000+numD,:);
d5 = DATA(60001:60000+numD,:);
d6 = DATA(75001:75000+numD,:);
D = [d1; d2; d3; d4; d5; d6];
DATA = D;


% Normalization [0-1]
DATA = normalize(DATA,'range');



% Parameters ========================================================
net.numNodes    = 0;   % the number of nodes
net.weight      = [];  % node position
net.CountNode = [];    % winner counter for each node
net.adaptiveSig = [];  % kernel bandwidth for CIM in each node
net.edge = zeros(2,2); % Initial connections (edges) matrix
net.LabelCluster = [];

net.Lambda = 100;       % an interval for calculating a kernel bandwidth for CIM
net.minCIM = 0.2;      % similarity threshold
% ==================================================================


time_train = 0;


for traial = 1:TRIAL*c
    
    fprintf('Iterations: %d/%d\n',traial,TRIAL*c);

    idx = mod(traial-1, c)+1;
    data = DATA(1+(numD*(idx-1)):numD*idx,:);
    
    % Noise Setting [0,1]
    if NR > 0
        noiseDATA = rand(size(data,1)*NR, size(data,2));
        data(1:size(noiseDATA,1),:) = noiseDATA;
    end
    
    % Randamize data
    rng(traial)
    ran = randperm(size(data,1));
    data = data(ran,:);
      
    % Training ==========================================
    tic
    net = ARTclustering_Train(data, net);
    time_train = time_train + toc;
    % ===================================================
    
    
    % Results
    resultNumNodes = ['   Num. Clusters:', num2str(net.numNodes)];
    disp(resultNumNodes);
    disp([' Processing Time:', num2str(time_train)]);
    disp('');

    
    % Create a figure with specified size
    figure(1);
    set(gcf, 'Position', [250, 200, 1200, 600]); % [left, bottom, width, height]
    set(gcf, 'Color', [1 1 1]);
    cla;

    % Plot network
    subplot(1, 2, 1);  % 1 row, 2 columns, first plot
    myPlot(data, net);

    % Plot estimated density
    subplot(1, 2, 2);  % 1 row, 2 columns, second plot
    estimateDensityByCountNode(net)
    axis square

    drawnow
    
end



