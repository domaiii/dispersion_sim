

clc; close all;

load wind_field2.mat   % expects: points (Nx2), cells (Mx3), u (Nx2)

points = double(points);
u      = double(u);
cells  = double(cells);

speed = compute_speed(u);
cmap  = sky(256);
cmin  = min(speed);
cmax  = max(speed);
if cmax == cmin, cmax = cmin + 1e-6; end

randomSeed = 5;

Nx = 30; Ny = 25;      % coarse grid resolution

% % === FIGURE 1: Original field ===
% % plot_original(points, u, cells, speed, cmap, cmin, cmax);
% 
% % === FIGURE 2: Random subset ===
% plot_subset(points, u, cells, speed, cmap, cmin, cmax, 35, randomSeed);
% 
% % === FIGURE 3: Coarse grid ===
% plot_coarse_grid(points, u, cells, cmap, cmin, cmax, Nx, Ny);

plot_subset_vs_coarse(points, u, cells, speed, cmap, cmin, cmax, ...
                      35, randomSeed, Nx, Ny);

%% ============================================================
%           FIGURE 1 — Original FULL resolution quiver
% =============================================================
function plot_original(points, u, cells, speed, cmap, cmin, cmax)
    figure('Color','w'); hold on; axis equal tight;
    title('Full resolution wind field','FontSize', 16)
    xlabel('x (m)', 'FontSize', 14)
    ylabel('y (m)', 'FontSize', 14)
    N = size(points,1);

    for k = 1:N
        col = speed_to_color(speed(k), cmin, cmax, cmap);
        quiver(points(k,1), points(k,2), u(k,1), u(k,2), ...
            0, 'Color', col, 'LineWidth', 0.8, 'MaxHeadSize', 0.7);
    end

    plot_boundaries(points, cells);
    
    colormap(cmap);
    cb = colorbar;
    cb.Label.String = 'Velocity (m/s)';
    cb.Label.FontSize = 14;
end



%% ============================================================
%       FIGURE 2 — Random subset of original DOFs
% =============================================================
function plot_subset(points, u, cells, speed, cmap, cmin, cmax, N_subset, seed)
    rng(seed);

    figure('Color','w'); hold on; axis equal tight;
    title(sprintf('Random Wind Samples'), 'FontSize',16)
    xlabel('x (m)', 'FontSize', 14)
    ylabel('y (m)', 'FontSize', 14)

    N = size(points,1);
    nPick = max(1, N_subset);
    idx = randperm(N, nPick);

    for k = idx
        col = speed_to_color(speed(k), cmin, cmax, cmap);
        quiver(points(k,1), points(k,2), u(k,1), u(k,2), ...
            0, 'Color', col, 'LineWidth', 1.5, 'MaxHeadSize', 0.7);
    end

    plot_boundaries(points, cells);

    colormap(cmap);
    cb = colorbar;
    cb.Label.String = 'Velocity (m/s)';
    cb.Label.FontSize = 14;
end



%% ============================================================
%           FIGURE 3 — Coarse Grid Visualization
% =============================================================
function plot_coarse_grid(points, u, cells, cmap, cmin, cmax, Nx, Ny)
    figure('Color','w'); hold on; axis equal tight;
    title('Coarse grid interpolated wind field','FontSize',16)
    xlabel('x (m)', 'FontSize', 14)
    ylabel('y (m)', 'FontSize', 14)
    xmin = min(points(:,1)); xmax = max(points(:,1));
    ymin = min(points(:,2)); ymax = max(points(:,2));

    [xq, yq] = meshgrid(linspace(xmin, xmax, Nx), ...
                        linspace(ymin, ymax, Ny));

    Fx = scatteredInterpolant(points(:,1), points(:,2), u(:,1), 'natural', 'none');
    Fy = scatteredInterpolant(points(:,1), points(:,2), u(:,2), 'natural', 'none');

    uqx = Fx(xq, yq);
    uqy = Fy(xq, yq);

    speed_q = sqrt(uqx.^2 + uqy.^2);

    valid = ~isnan(uqx) & ~isnan(uqy);
    idx = find(valid);

    for ii = idx'
        col = speed_to_color(speed_q(ii), cmin, cmax, cmap);
        quiver(xq(ii), yq(ii), uqx(ii), uqy(ii), ...
            'Color', col, 'LineWidth', 1.0, 'MaxHeadSize', 0.7, 'AutoScaleFactor', 1.0);
    end

    plot_boundaries(points, cells);

    colormap(cmap);
    cb = colorbar;
    cb.Label.String = 'Velocity (m/s)';
    cb.Label.FontSize = 14;
end



%% ============================================================
%                HELPER FUNCTIONS
% =============================================================
function s = compute_speed(u)
    s = sqrt(u(:,1).^2 + u(:,2).^2);
end


function col = speed_to_color(val, cmin, cmax, cmap)
    idx = round((val - cmin) / (cmax - cmin) * (size(cmap,1)-1)) + 1;
    idx = max(min(idx, size(cmap,1)), 1);
    col = cmap(idx, :);
end


function plot_boundaries(points, cells)
    TR = triangulation(cells, points);
    be = freeBoundary(TR);

    for i = 1:size(be,1)
        p = be(i,:);
        plot(points(p,1), points(p,2), 'k-', 'LineWidth', 1.3);
    end
end

function plot_subset_vs_coarse(points, u, cells, speed, cmap, cmin, cmax, ...
                               N_subset, seed, Nx, Ny)

    figure('Color','w');
    t = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    % Top panel
    ax1 = nexttile; hold(ax1,'on'); axis(ax1,'equal'); axis(ax1,'tight');
    title(ax1, 'Random Wind Samples', 'FontSize',16);
    ax1.FontSize = 14;
    xlabel(ax1, 'x (m)', 'FontSize',14);
    ylabel(ax1, 'y (m)', 'FontSize',14);

    N = size(points,1);
    rng(seed);
    idx = randperm(N, min(N_subset, N));

    for k = idx
        col = speed_to_color(speed(k), cmin, cmax, cmap);
        quiver(ax1, points(k,1), points(k,2), u(k,1), u(k,2), 0, ...
            'Color', col, 'LineWidth', 1.4, 'MaxHeadSize', 0.7);
    end

    plot_boundaries(points, cells);

    colormap(cmap);
    cb1 = colorbar(ax1);
    cb1.Label.String = 'Velocity (m/s)';
    cb1.Label.FontSize = 14;
    caxis(ax1, [cmin cmax]);


    %  Bottom panel
    ax2 = nexttile; hold(ax2,'on'); axis(ax2,'equal'); axis(ax2,'tight');
    title(ax2, 'Estimated Wind Field', 'FontSize',16);
    ax2.FontSize = 14;
    xlabel(ax2, 'x (m)', 'FontSize',14);
    ylabel(ax2, 'y (m)', 'FontSize',14);  
    xmin = min(points(:,1)); xmax = max(points(:,1));
    ymin = min(points(:,2)); ymax = max(points(:,2));

    [xq, yq] = meshgrid(linspace(xmin, xmax, Nx), linspace(ymin, ymax, Ny));

    Fx = scatteredInterpolant(points(:,1), points(:,2), u(:,1), 'natural','none');
    Fy = scatteredInterpolant(points(:,1), points(:,2), u(:,2), 'natural','none');

    uqx = Fx(xq, yq);
    uqy = Fy(xq, yq);

    speed_q = sqrt(uqx.^2 + uqy.^2);
    valid = ~isnan(uqx) & ~isnan(uqy);
    idx_q = find(valid);

    for ii = idx_q'
        col = speed_to_color(speed_q(ii), cmin, cmax, cmap);
        quiver(ax2, xq(ii), yq(ii), uqx(ii), uqy(ii), 0, ...
            'Color', col, 'LineWidth', 1.1, 'MaxHeadSize', 0.7);
    end

    plot_boundaries(points, cells);

    cb2 = colorbar(ax2);
    cb2.Label.String = 'Velocity (m/s)';
    cb2.Label.FontSize = 14;
    caxis(ax2, [cmin cmax]);
    
    % make layout as tight as possible
    t.TileSpacing = 'compact';
    t.Padding = 'compact';
    drawnow;
    
    % reduce axes loose inset (optional)
    for ax = [ax1, ax2]
        inset = get(ax, 'TightInset');              % [left bottom right top]
        set(ax, 'Position', [ax.Position(1:2), ax.Position(3:4)]); % keep current pos
        % optional: reduce loose spacing
        set(ax, 'LooseInset', max(inset, 0.02));
    end
    
    filename = 'windplot.pdf';
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

end