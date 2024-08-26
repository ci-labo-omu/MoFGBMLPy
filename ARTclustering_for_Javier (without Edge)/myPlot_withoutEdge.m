function myPlot_withoutEdge(DATA, net, num)

cla;

w = net.weight;
[N,D] = size(w);

label = net.LabelCluster;

% figure(num);

hold on;

% データポイントをプロット
plot(DATA(:,1), DATA(:,2), 'o', 'MarkerEdgeColor', [0.8, 0.8, 0.8], 'MarkerFaceColor', [0.7, 0.7, 0.7], 'MarkerSize', 3);


% ノードを描画
color = [
    [1 0 0]; 
    [0 1 0]; 
    [0 0 1]; 
    [1 0 1];
    [0.8500 0.3250 0.0980];
    [0.9290 0.6940 0.1250];
    [0.4940 0.1840 0.5560];
    [0.4660 0.6740 0.1880];
    [0.3010 0.7450 0.9330];
    [0.6350 0.0780 0.1840];
];
m = length(color);

for k = 1:N
    if D == 2
        plot(w(k,1), w(k,2), '.', 'Color', color(mod(label(1,k)-1,m)+1,:), 'MarkerSize', 35);
    elseif D == 3
        plot3(w(k,1), w(k,2), w(k,3), '.', 'Color', color(mod(label(1,k)-1,m)+1,:), 'MarkerSize', 35);
    end
end

% CountNodeを各ノードの右上に描画
for i = 1:N
    countStr = num2str(net.CountNode(i));  % CountNodeを文字列に変換
    if D == 2
        text(w(i,1) + 0.01, w(i,2) + 0.01, countStr, 'Color', 'k', 'FontSize', 8);
    elseif D == 3
        text(w(i,1) + 0.01, w(i,2) + 0.01, w(i,3) + 0.01, countStr, 'Color', 'y', 'FontSize', 14);
    end
end

ytickformat('%.1f');
xtickformat('%.1f');


set(gca,'GridColor','k')
set(gca,'layer','bottom');

axis equal
grid on
box on
hold off
axis([0 1 0 1]); %ignore
ytickformat('%.1f') %ignore
xtickformat('%.1f') %ignore
xlabel('X',  'FontSize', 14);
ylabel('Y',  'FontSize', 14);
% pause(0.01); %ignore

end
