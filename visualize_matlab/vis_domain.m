clc
close all
clear all

load domain.mat
cells = double(cells);
points = double(points);
facets = double(facets);
facet_indices = double(facet_indices);
facet_values = double(facet_values);

TR = triangulation(cells, points);
figure; hold on; axis equal tight
triplot(TR, 'Color', [0.85 0.85 0.85]);
title('Experiment Domain', 'FontSize', 16);
xlabel('x (m)', 'FontSize', 13)
ylabel('y (m)', 'FontSize', 13)

colors = containers.Map('KeyType','double','ValueType','any');
colors(2) = [1 0 0];    % Inflow
colors(3) = [0 1 0];  % Outflow
colors(4) = [0 0 1];% Wall / no-slip
colors(5) = [0 0 1];      % Obstacle
defaultColor = [0.5 0.5 0.5];

labels_added = containers.Map('KeyType','double','ValueType','any');

for k = 1:length(facet_indices)
    fi = facet_indices(k);
    tag = facet_values(k);
    xy = points(facets(fi,:), :);

    col = defaultColor;
    if isKey(colors, tag)
        col = colors(tag);
    end

    plot(xy(:,1), xy(:,2), 'Color', col, 'LineWidth', 3);

    % Add legend only once per tag
    if ~isKey(labels_added, tag)
        h = plot(nan, nan, 'Color', col, 'LineWidth', 3);
        labels_added(tag) = h;
    end
end

% Legend
legendHandles = values(labels_added);
legendTags    = keys(labels_added);
legendEntries = arrayfun(@(t) tag2name(t), cell2mat(legendTags), 'UniformOutput', false);

legend([legendHandles{:}], legendEntries, 'Location','bestoutside');

% --- Parabolic inflow profile (Tag = 2) ---
inflow_facets = facet_indices(facet_values == 2);

if ~isempty(inflow_facets)
    inflow_pts = unique(facets(inflow_facets, :));
    xy_inflow  = points(inflow_pts, :);

    % Sort by y-coordinate
    [~, order] = sort(xy_inflow(:,2));
    xy_inflow = xy_inflow(order, :);

    % Parabolic profile parameters
    Umax = 10.0;                  
    x0 = min(points(:,1));      

    ymin = min(xy_inflow(:,2));
    ymax = max(xy_inflow(:,2));
    H = ymax - ymin;
    yc = (ymax + ymin)/2;

    % Parabolic velocity
    u_prof = Umax * (1 - ((xy_inflow(:,2)-yc)/(H/2)).^2);

    % Plot arrows
    scale = 0.12;  
    for i = 1:length(u_prof)
        quiver(x0, xy_inflow(i,2), scale*u_prof(i), 0, ...
            'Color', [1 0 0], 'LineWidth', 1.8, 'MaxHeadSize', 2, ...
            'HandleVisibility', 'off');
    end
end

hold off


function name = tag2name(tag)
switch double(tag)
    case 2, name = 'Inflow';
    case 3, name = 'Outflow';
    case 4, name = 'Wall';
    case 5, name = 'Obstacles';
    otherwise, name = sprintf('Tag %d', double(tag));
end
end

