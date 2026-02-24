clc
close all

load domain.mat
cells = double(cells);
points = double(points);
facets = double(facets);
facet_indices = double(facet_indices);
facet_values = double(facet_values);

TR = triangulation(cells, points);
figure; hold on; axis equal tight
triplot(TR, 'Color', [0.85 0.85 0.85]);

colors = containers.Map('KeyType','char','ValueType','any');
colors('Inflow')  = [0.3 0.5 0.85];
colors('Outflow') = [0.6 0.6 0.6];
colors('Walls/Obstacles') = [0.1 0.1 0.1];
defaultColor = [0.5 0.5 0.5];

labels_added = containers.Map('KeyType','char','ValueType','any');

for k = 1:length(facet_indices)
    fi = facet_indices(k);
    tag = facet_values(k);
    xy = points(facets(fi,:), :);
    name = tag2name(tag);
    name = char(name);
    col = defaultColor;
    if isKey(colors, name)
        col = colors(name);
    end

    if tag == 3       % Outflow
        ls = '--';    % dashed
    else
        ls = '-';
    end
    lw = 2;

    plot(xy(:,1), xy(:,2), 'Color', col, 'LineWidth', lw, 'LineStyle', ls);

    % --- In der Schleife nach dem plot(xy(:,1), ...) ---
    name = tag2name(tag);
    
    % Add legend only once per name (not per tag)
    if ~isKey(labels_added, name)
        h = plot(nan, nan, 'Color', col, 'LineWidth', 3);
        labels_added(name) = h;
    end
end

% Legend
legendNames   = keys(labels_added);
legendHandles = values(labels_added);
legendEntries = legendNames; % already strings

legend([legendHandles{:}], legendEntries, 'Location','northeast', 'FontSize', 15);

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
            'Color', [0.3 0.5 0.85], 'LineWidth', 1.8, 'MaxHeadSize', 2, ...
            'HandleVisibility', 'off');
    end
end

% title('Experiment Domain', 'FontSize', 18);
ax = gca;
ax.FontSize = 13;
xlabel('x (m)', 'FontSize', 15)
ylabel('y (m)', 'FontSize', 15)

% reduce axes loose inset (optional)
inset = get(ax, 'TightInset');              % [left bottom right top]
set(ax, 'Position', [ax.Position(1:2), ax.Position(3:4)]); % keep current pos
% optional: reduce loose spacing
set(ax, 'LooseInset', max(inset, 0.02));


filename = 'domain_vis.pdf';
exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');
hold off



function name = tag2name(tag)
switch double(tag)
    case 2, name = 'Inflow';
    case 3, name = 'Outflow';
    case 4, name = 'Walls/Obstacles';
    case 5, name = 'Walls/Obstacles';
    otherwise, name = sprintf('Tag %d', double(tag));
end
end

