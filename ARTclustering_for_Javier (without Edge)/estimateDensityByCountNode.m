function estimateDensityByCountNode(net)
    
    % cla;

    % Retrieve node positions and counts
    node_positions = net.weight;
    count_node = net.CountNode;

    % Bandwidth calculation using Silverman's Rule
    n = size(node_positions, 1);
    sigma_x = std(node_positions(:, 1));
    sigma_y = std(node_positions(:, 2));
    
    h_x = sigma_x * (4 / (3 * n))^(1/5);
    h_y = sigma_y * (4 / (3 * n))^(1/5);

    % Set the grid range and resolution
    grid_size = 100; % Grid resolution
    x_grid = linspace(0, 1, grid_size);
    y_grid = linspace(0, 1, grid_size);
    
    [X, Y] = meshgrid(x_grid, y_grid);
    density = zeros(size(X));
    
    % Calculate density for each grid point
    for i = 1:grid_size
        for j = 1:grid_size
            % Coordinates of the grid point
            grid_point = [X(i,j), Y(i,j)];
            
            % Calculate kernel density for each node
            distances_x = (node_positions(:, 1) - grid_point(1)) / h_x;
            distances_y = (node_positions(:, 2) - grid_point(2)) / h_y;
            kernel_values = count_node' .* exp(-(distances_x.^2 + distances_y.^2) / 2);
            
            % Accumulate density
            density(i,j) = sum(kernel_values);
        end
    end
    
    % Normalize density to the range [0, 1]
    density = density / max(density(:));
    
    % Plot the 3D surface
    % figure(num);
    surf(X, Y, density);  % 3D surface plot
    colormap(jet); % Set colormap to jet
    alpha(0.8);
    
    % Configure the colorbar
    c = colorbar;
    clim([0 1]);
    c.Ticks = 0:0.1:1.0; % Set ticks at 0.0, 0.2, ..., 1.0
    c.TickLabels = cellstr(num2str(c.Ticks', '%.1f')); % Display tick labels with one decimal place
    
    % Set X, Y, Z axis ticks
    xticks(0.0:0.2:1.0);
    yticks(0.0:0.2:1.0);
    zticks(0.0:0.2:1.0);

    % Set X, Y, Z axis tick labels
    xticklabels(cellstr(num2str((0.0:0.2:1.0)', '%.1f')));
    yticklabels(cellstr(num2str((0.0:0.2:1.0)', '%.1f')));
    zticklabels(cellstr(num2str((0.0:0.2:1.0)', '%.1f')));

    % Set X, Y, Z axes to the range [0, 1]
    xlim([0.0 1.0]);
    ylim([0.0 1.0]);
    zlim([0.0 1.0]);
    
    title('Estimated Density by CountNode', 'FontSize', 14);
    xlabel('X',  'FontSize', 14);
    ylabel('Y',  'FontSize', 14);
    zlabel('Density', 'FontSize', 14);
    view(3); % Set 3D view
    
end
