function runWhaleDataVisualizer()
% This function provides a visualization environment for acoustic whale data.
    
    % background variables
    background_file = 'background_spectrograms_200_50.csv';
    spectrogram_size = [200, 50];
    background_cutoff = 0.75;

    % ping parameters
    period = 3998.81;
    ping_length = 449;
    ping_left_shift = 20;


    fprintf('-------------- WHALE DATA VISUALIZER -------------- \n');
    fprintf('Current datasets:\n');
    fprintf('1: dataset from 2014 / 02-14-11-42 \n');
    fprintf('2: dataset from 2014 / 02-14-09-53 \n');
    fprintf('3: dataset from 2014 / 02-10-08-17 \n\n');
    num = input('Please enter 1, 2, or 3 to use one of the following datasets: ');
    
    [data, header, info, Data, subdata, subdata_start, subdata_length, exact_loc] = chooseDataset(num, ping_left_shift);
    
    [~, ~, V] = performBackgroundSVD(background_file, spectrogram_size);
    
    [data_segments, data_inds, num_windows] = makeWindows( exact_loc, period, ping_length, subdata );
    
    E = findAllEventsInWindows(Data.Events, info, subdata_start, data_inds, period, round(period - ping_length), spectrogram_size);
    fprintf('\nThere are %d events inside of the windowed data segments.\n', size(E, 2));
    
    K = calculateSpectrogramsOverWindows(data_segments, info, num_windows, spectrogram_size);
    
    M = subtractBackgroundComponents(K, V, num_windows, background_cutoff);
    
    M4 = connectedComponentsFiltering(M, spectrogram_size, 20, 20);
    
    
    % VISUALIZATION LOOP
    fprintf('\n---------- Visualization ------------\n');
    while (1)
        start_at = input(sprintf('Please enter a starting window index between 1 and %d. Enter 0 to quit. ', (num_windows - 20 + 1)));
        if (start_at == 0)
            break;
        end
        if (start_at < 1 || start_at > num_windows)
            start_at = input(sprintf('Starting window index must be between 1 and %d. Please re-enter. ', (num_windows - 20 + 1)));
            
            if (start_at == 0)
                break;
            end
        end
        
        spectype = input('Which spectograms? Please enter: 1 for original, 2 for background-subtracted, or 3 for fully segmented. ');
        if (spectype ~= 1 && spectype ~= 2 && spectype ~= 3)
            spectype = input('Please enter either 1, 2, or 3. ');
        end
        
        specmatrix = K;
        climits = [5 14];
        titlestring = 'Original Spectrograms';
        if (spectype == 1)
            specmatrix = K;
            climits = [5 14];
            col = 'black';
            titlestring = 'Original Spectrograms';
        elseif (spectype == 2)
            specmatrix = M;
            climits = [0 3];
            col = 'white';
            titlestring = 'Background-subtracted Spectrograms';
        elseif (spectype == 3)
            specmatrix = M4;
            climits = [0 3];
            col = 'white';
            titlestring = 'Fully Segmented Spectrograms';
        else
            fprintf('Input must be 1, 2, or 3. Defaulting to original.\n');
        end
            
        plotSpec(specmatrix, 10, start_at, spectrogram_size, E, climits, titlestring, col);
        fprintf('\n');
    end
end

% -------------------------------------------------------------------------
% PLOT SPEC
%
% This function takes a matrix of spectrograms and plots them, boxing any
% events that are present.
%
% INPUT:
%   M -- matrix of spectrograms
%   num_plots -- number of spectrograms to plot
%   start_at -- starting row index which to plot
%   spectrogram_size -- 2d spectrogram size
%   E -- event structure as made by findAllEventsInWindows
%   caxis_limits -- limits for the color axis (for consistency across plots)
%   titlestring -- string representing spectrogram title
%   col -- box/label color
%
% OUTPUT: figure.
%
function plotSpec( M, num_plots, start_at, spectrogram_size, E, caxis_limits, titlestring, col)

    figure('position', [0, 500, 1500, 400])
    for j = 1:num_plots
        subplot(1, num_plots, j)
        j = j + start_at - 1;
        if (j > size(M, 1))
            break
        end

        pcolor(reshape(M(j, :), spectrogram_size)), shading interp, colormap(jet)
        set(gca,'YTickLabel',[],'XTickLabel',[]); xlabel(sprintf('%d', j));

        caxis([caxis_limits])

        hold on
        for jj = 1:size(E, 2)
            if (E{jj}.window == j)
                %fprintf('EVENT %d -- Call type: %s \n', jj, E{jj}.call_type);

                rectangle('Position', [E{jj}.time1, E{jj}.freq1,...
                    (E{jj}.time2 - E{jj}.time1), (E{jj}.freq2 - E{jj}.freq1)], ...
                'EdgeColor', col);
                text(E{jj}.time1, E{jj}.freq2 + 5, E{jj}.call_type, 'Color', col);
            end
        end
        hold off
    end

    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(0.5, 0.97, sprintf('%s: %d to %d', titlestring, start_at, start_at + num_plots - 1),...
        'HorizontalAlignment','center','VerticalAlignment', 'middle', 'FontSize', 16,...
        'FontWeight', 'bold');
end


% -------------------------------------------------------------------------
% CHOOSE DATASET
%
% This function does the interactive dataset selection.
%
% INPUT:
%   num -- dataset index
%   ping_left_shift -- shift to apply to ping index
%
% OUTPUT:
%   data, header, info -- data resulting from MTRead
%   Data -- event information in .mat file
%   subdata -- specific subset of data
%   subdata_start -- index of subdata within data
%   subdata_length -- length of subdata (in number of samples)
%   exact_loc -- exact position of a reference ping in subdata
%
function [data, header, info, Data, subdata, subdata_start, subdata_length, exact_loc] = chooseDataset(num, ping_left_shift)
    if (num == 1)
        fprintf('Loading dataset 2014 / 02-14-11-42... \n');
        load('MT_Data/cb_LoPower_2014_02141142-notes-kent-revised3.mat');
        [data, header, info] = MTRead('MT_Data/cb_LoPower_2014_02141142.mt');
        subdata_start = 1.9e7;
        subdata_length = 500000;
        approx_loc = [4380, 4880];
        
    elseif (num == 2)
        fprintf('Loading dataset 2014 / 02-14-09-53... \n');
        load('MT_Data/cb_LoPower_2014_02140953-notes-kseger-revised3.mat');
        [data, header, info] = MTRead('MT_Data/cb_LoPower_2014_02140953.mt');
        subdata_start = 36100001;
        subdata_length = 500000;
        approx_loc = [50600, 51200];
    elseif (num == 3)
        fprintf('Loading dataset 2014 / 02-10-08-17... \n');
        load('MT_Data/CB_LoPower_2014_02100817-notes-p1fernan-revised3.mat');
        [data, header, info] = MTRead('MT_Data/CB_LoPower_2014_02100817.mt');
        subdata_start = 2e7;
        subdata_length = 500000;
        exact_loc = 40222;
    else
        x = input('Please enter 1, 2, or 3.');
        [data, header, info, Data, subdata, subdata_start, subdata_length, approx_loc] = chooseDataset(x, ping_left_shift);
    end
    
    subdata = data(subdata_start : subdata_start+subdata_length-1)';
    subdata = detrend(subdata);
    subdata = subdata - mean(subdata);
    
    if (num == 1 || num == 2)
        exact_loc = findExactPingLocation(subdata, approx_loc, ping_left_shift);
    end
end

% -------------------------------------------------------------------------
% FIND EXACT PING LOCATION
function [exact_loc] = findExactPingLocation(subdata, approx_loc, ping_left_shift)
    
    ping_area = subdata(approx_loc(1):approx_loc(2));
    ping_area = ping_area - mean(ping_area);

    % look for the first ping
    [pks, locs, w, p] = findpeaks(ping_area, 'MinPeakHeight', 5000,...
        'MinPeakProminence', 8000, 'NPeaks', 1);

    if (size(locs, 2) ~= 1)
        error('error! wrong number of locs!');
    end

    locs = locs + approx_loc(1) - 1;
    exact_loc = locs - ping_left_shift;
end


% -------------------------------------------------------------------------
% PERFORM BACKGROUND SVD
%
% This function performs PCA on the background spectrogram file. Uncomment
% code for figure of components and singular values.
%
% INPUT:
%   background_file -- name of file containing background spectrograms
%   spectrogram_size -- 2d size of spectrograms
%
% OUTPUT:
%   U, S, V -- singular value decomposition of the given file
%
function [U, S, V] = performBackgroundSVD(background_file, spectrogram_size)
    fprintf('\nDecomposing background into components...\n');
    
    % to create background components, run making_specs_around_background
    X = csvread(background_file);

    % compute SVD and coefficients (YY)
    [U, S, V] = svd(X, 'econ');
    Y = U * S;
    svalues = diag(S);

    % plot components of signal
%     fprintf('-- plotting background components\n');
%     figure
%     for j = 1:9
%         subplot(3, 3, j);
%         pcolor(reshape(V(:, j), spectrogram_size)), shading interp;
%         set(gca, 'Fontsize', 13)
%         xlabel('Time (sec)'), ylabel('Frequencies');
%         title(sprintf('SVD Mode %d of Signal', j))
%         colormap(jet);
%     end
%     drawnow
    
    % plot singular values
%     fprintf('-- plotting singular values of components\n');
%     figure
%     plot(sort(svalues, 'descend')/max(svalues), '*')
%     xlabel('Index of singular value'); ylabel('Proportion of energy in mode');
%     axis([1, 20, 0, 1.1])
%     drawnow
end

% -------------------------------------------------------------------------
% PERFORM BACKGROUND SVD
%
% This function performs PCA on the background spectrogram file. Uncomment
% code for figure of components and singular values.
%
% INPUT:
%   background_file -- name of file containing background spectrograms
%   spectrogram_size -- 2d size of spectrograms
%
% OUTPUT:
%   U, S, V -- singular value decomposition of the given file
%
function [data_segments, data_inds, num_windows] = makeWindows( exact_loc, period, ping_length, subdata )
    % Compute the ping-free windows
    fprintf('\nSplitting data into windows...\n');

    % figure out number of windows and location
    window_length = round(period - ping_length - 1);
    first_ping = round(exact_loc - period * floor(exact_loc / period));

    if (first_ping - ping_length > window_length)
        first_ping = round(first_ping - period);
    end

    num_windows = floor((size(subdata, 2) - (first_ping))/period);

    % save segments between pings
    data_segments = zeros(num_windows, round(period - ping_length));
    data_inds = zeros(num_windows, 2);

    for j = 1:num_windows
        start_ind = round(period * (j-1) + first_ping + ping_length);
        end_ind = round(start_ind + period - ping_length - 1);
        data_inds(j, :) = [start_ind, end_ind - 1];

        data_segments(j, :) = subdata(start_ind:end_ind);
    end
    
    % Plot windows relative to ping locations
%     fprintf('-- plotting window locations over raw audio signal \n');
%     fprintf('-- (red is window start, green is window end) \n');
%     figure
%     plot(subdata)
%     for j = 1:num_windows
%         line([ data_inds(j, 1), data_inds(j, 1)], [-30000, 30000], 'Color', 'r');
%         line([ data_inds(j, 2), data_inds(j, 2)], [-30000, 30000], 'Color', 'g');
%     end
%     xlabel('Sample index'); ylabel('Amplitude');
%     set(gca, 'FontSize', 13);
%     drawnow
end


% -------------------------------------------------------------------------
% CALCULATE SPECTROGRAMS OVER WINDOWS
%
% This function calculates spectrograms for each window in a data_segments
% matrix and saves it as a row in a matrix K.
%
% INPUT:
%   data_segments -- matrix where each row contains raw data from one window
%   info -- info object, just needed for sampling rate
%   num_windows -- total number of windows (rows in data_segments)
%   spectrogram_size -- 2d size of spectrograms
%
% OUTPUT:
%   K -- matrix where each row is a spectrogram
%
function [K] = calculateSpectrogramsOverWindows(data_segments, info, num_windows, spectrogram_size)
    fprintf('\nCalculating spectrograms over windows...\n');
    % make matrix for holding spectrograms from this specific dataset
    K = [];

    % make a spectrogram for each window
    tic
    for j = 1:num_windows
        % Take these points from the dataset
        dat = data_segments(j, :);
        dat = detrend(dat);
        dat = dat - mean(dat);

        % Compute a spectrogram (and print the time taken for the computation)
        % Look in the spectro_smooth function for a description of parameters.
        [ spec, t, k ] = spectro_smooth(dat, info.srate, 400, spectrogram_size(2), 3, -1);

        % Take the logarithm
        logspec = log(spec.');

        % Decrease spectrogram size (frequency axis)
        n = spectrogram_size(1);
        k = linspace(min(k), max(k), n);
        logspec = average_down(logspec, n, floor(size(spec, 2) / n));

        % Add to matrix holding spectrograms
        K = [K; logspec(:).']; % holds not smoothed spectrograms

        if (mod(j, 500) == 0)
            fprintf('Finished window number %d \n', j);
        end
    end
    t = toc;
    fprintf('Spectrograms computation took %.3f sec.\n', t);
end


% -------------------------------------------------------------------------
% SUBTRACT BACKGROUND COMPONENTS
%
% This function calculates spectrograms for each window in a data_segments
% matrix and saves it as a row in a matrix K.
%
% INPUT:
%   K -- matrix where each row is a spectrogram
%   V -- V matrix from SVD
%   num_windows -- total number of windows (rows in data_segments)
%   background_cutoff -- value under which to zero out spectrograms values
%
% OUTPUT:
%   M -- matrix where each row is a background-subtracted spectrogram
%
function M = subtractBackgroundComponents(K, V, num_windows, background_cutoff)
    fprintf('\nSubtracting background from spectrograms...\n');    
    M = zeros(size(K));
    tic
    for j = 1:num_windows
        m = K(j,:) - (K(j,:) * V(:, 1:2)) * V(:, 1:2)';

        m(m < background_cutoff) = 0;

        M(j, :) = m;
    end
    t = toc;
    fprintf('Background subtraction took %.3f sec.\n', t);
end

function [all_events_struct] = findAllEventsInWindows(events, info, subdata_start, data_inds,...
    period, window_length, spectrogram_size)
% This function takes a Data.Events object and data parameters and returns
% a new structure relating the events to the data windows.
    
    all_events_struct = {};
    for j = 1:size(events, 1)
        event_struct = findEventInWindows(events(j), info, subdata_start, data_inds,...
            period, window_length, spectrogram_size);
        if (isstruct(event_struct))
            all_events_struct = [all_events_struct, event_struct];
        end
    end
end


% This function takes an event (such as Data.Events(3)) and data parameters
% and creates a structure that holds information about the window that this
% event is in along with the pixel coordinates (in time and frequency).
function [event_struct] = findEventInWindows(event, info, subdata_start, data_inds,...
    period, window_length, spectrogram_size)

    event_ind = (event.start_time - info.datenumber) * 24 * 3600 * info.srate - subdata_start;
    event_dur = str2double(event.duration) * info.srate;
  
    window = ceil((event_ind - data_inds(1))/period);
    num_windows = size(data_inds, 1);
    
    if (window > 0 && window < num_windows && ...
            event_ind >= data_inds(window, 1) && event_ind < data_inds(window, 2))
        time1 = (event_ind - data_inds(window, 1)) / window_length * spectrogram_size(2);
        time2 = time1 + event_dur / window_length * spectrogram_size(2);
        freq1 = str2double(event.min_freq) * 2/info.srate * spectrogram_size(1);
        freq2 = str2double(event.max_freq) * 2/info.srate * spectrogram_size(1);
        event_struct = struct('window', window, 'time1', time1, 'time2', time2,...
    'freq1', freq1, 'freq2', freq2, 'call_type', event.call_type);
    else
        event_struct = 0;
    end
end

function [ M4 ] = connectedComponentsFiltering( M, spectrogram_size, pixel_num_thresh, pixel_power_thresh )
% This function accepts a matrix of (background-subtracted) spectrograms and 
% removes connected components that have less than the threshold number of
% pixels and power.
%
% INPUT:
%   M -- matrix in which each row is a (sparse) spectrogram
%   spectrogram_size -- 2D size of spectrograms in M
%   pixel_num_thresh -- the minimum required pixel count of a component
%   pixel_power_thresh -- the minimum required power of a component
%
% OUTPUT:
%   M4 -- matrix of spectrograms with some components removed

    fprintf('\nAdditional filtering of spectrograms with connected components... \n');
    num_windows = size(M, 1);
    
    tic
    M4 = zeros(size(M));
    for j = 1:num_windows  
        % take a window
        m = reshape(M(j, :), spectrogram_size);

        % cut out noise in the bottom 0hz-200hz
        m(1:10, :) = 0;

        % find connected components
        CC = bwconncomp(m);

        % calculate power and number of each component
        pixelPower = cellfun(@(l) sum(m(l)), CC.PixelIdxList);
        pixelNum = cellfun(@numel, CC.PixelIdxList);

        % remove connected components containing fewer than 8 pixels
        [idx] = find(pixelNum < pixel_num_thresh & pixelPower < pixel_power_thresh);
        m(vertcat(CC.PixelIdxList{idx})) = 0;

        % put into row of M4
        M4(j, :) = m(:);
    end
    t = toc;
    fprintf('Connected component analysis took %.3f sec\n', t);

end

function [ small_spec ] = average_down( spec, num_points, num_smooth )
% Average a matrix together in the y direction to make it smaller and
% keep it smooth. Assume spec is of size (frequency) x (time).
    
    total_points = num_points * num_smooth;
    
    if (total_points > size(spec, 1))
        error('num_points * num_smooth must be less than or equal to the number of rows in spec');
    end
    
    small_spec = zeros(num_points, size(spec, 2));
    point_indeces = floor((1:total_points)/total_points * size(spec, 1));

    for j = 1:num_points
        from = point_indeces((j - 1) * num_smooth + 1);
        to = point_indeces(j * num_smooth);
        small_spec(j, :) = mean(spec(from:to, :), 1);
    end
end

function [ spec, t, k ] = spectro_smooth( S, Fs, hamm, num_points, num_smooth, res)
    % This function computes a (trimmed, averaged) spectrogram of a signal.
    % INPUT:
    %   S -- input signal (vector of length n)
    %   Fs -- sampling rate (points per second)
    %   hamm -- number of points in Hanning window
    %   num_points -- number of time points in the final spectrogram
    %   num_smooth -- how many consecutive steps to average for each time point
    %   res -- the number of lower frequencies to take (or -1 for all frequencies)
    % OUTPUT:
    %   smoothspec -- the (averaged, trimmed) spectrogram
    %   t -- the time vector
    %   k -- the frequency vector
    
    % Create vector for frequencies
    n = length(S); L = n / Fs;
    %ks = (2 * pi/L) * [-n/2 : n/2-1]; k = fftshift(ks);

    % Create a sliding window. At each time step, multiply the signal by a
    % Gaussian, take the frequency content, and append it to the spectrogram.
    g = hann(hamm).';

    point_indeces = floor((1:num_points * num_smooth) * (n - hamm)/(num_points * num_smooth));

    fullspec = zeros(num_points, n);
    for j = 1:num_points
        buffer = zeros(num_smooth, n); % buffer storing spectra to average
        for jj = 1:num_smooth
            point_ind = point_indeces((j - 1) * num_smooth + jj);
            Sg = g .* S(point_ind : point_ind + hamm - 1);
            Sg = [zeros(1, point_ind - 1), Sg, zeros(1, n - point_ind - hamm + 1)];
            Sgt = fft(Sg);
            buffer(jj, :) = abs(Sgt);
        end
        fullspec(j,:) = mean(buffer, 1);
    end
    
    k = Fs * (0:(n/2))/n;
    spec = fullspec(:, 1:length(k));
    
    % Trim to contain lower frequencies
    if (res > 0)
        spec = fullspec(:, 1:res);
        k = k(1:res);
    end
    
    % New time vector
    t2 = linspace(0, L, num_points + 1); t = t2(1:num_points);
end

