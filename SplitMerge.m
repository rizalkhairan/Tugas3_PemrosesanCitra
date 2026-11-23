
classdef SplitMerge
    methods (Static)
        function labels = segment(I, splitThreshold, mergeThreshold, minSize)
            if size(I, 3) == 3
                I = rgb2gray(I);
            end
            I = im2double(I);

            validateattributes(splitThreshold, {'numeric'}, {'scalar', 'nonnegative', 'finite'});
            validateattributes(mergeThreshold, {'numeric'}, {'scalar', 'nonnegative', 'finite'});
            validateattributes(minSize, {'numeric'}, {'scalar', 'integer', 'positive'});

            % Convert UI thresholds from [0, 255] to [0, 1]
            splitThreshold = splitThreshold / 255;
            mergeThreshold = mergeThreshold / 255;

            % Split
            labels = zeros(size(I), 'uint32');
            nextLabel = 1;

            [labels, ~] = SplitMerge.split( ...
                I, labels, 1, size(I, 1), ...
                1, size(I, 2), ...
                splitThreshold, minSize, nextLabel);

            % Merge
            labels = SplitMerge.merge(I, labels, mergeThreshold);
        end
    end

    methods (Static, Access = private)

        function val = Q(region, threshold)
            % Homogeneity predicate: standard deviation
            val = std(region(:), 1) <= threshold;
        end

        function [labels, nextLabel] = split( ...
                I, labels, r1, r2, c1, c2, ...
                threshold, minSize, nextLabel)

            region = I(r1:r2, c1:c2);

            height = r2 - r1 + 1;
            width = c2 - c1 + 1;

            if SplitMerge.Q(region, threshold) || ...
                    height < 2 * minSize || ...
                    width < 2 * minSize

                labels(r1:r2, c1:c2) = nextLabel;
                nextLabel = nextLabel + 1;
                return;
            end

            % Find midpoint
            rm = floor((r1 + r2) / 2);
            cm = floor((c1 + c2) / 2);

            % Top-left
            [labels, nextLabel] = SplitMerge.split( ...
                I, labels, r1, rm, c1, cm, ...
                threshold, minSize, nextLabel);

            % Top-right
            [labels, nextLabel] = SplitMerge.split( ...
                I, labels, r1, rm, cm + 1, c2, ...
                threshold, minSize, nextLabel);

            % Bottom-left
            [labels, nextLabel] = SplitMerge.split( ...
                I, labels, rm + 1, r2, c1, cm, ...
                threshold, minSize, nextLabel);

            % Bottom-right
            [labels, nextLabel] = SplitMerge.split( ...
                I, labels, rm + 1, r2, cm + 1, c2, ...
                threshold, minSize, nextLabel);
        end

        function labels = merge(I, labels, threshold)
            n = double(max(labels(:)));

            areas = accumarray(double(labels(:)), 1, [n, 1]);
            sums = accumarray(double(labels(:)), I(:), [n, 1]);

            parent = (1:n)';

            pairs = SplitMerge.getAdjacentPairs(labels);

            % Iteratively merge most similar adjacent regions
            while ~isempty(pairs)

                for k = 1:size(pairs, 1)
                    pairs(k, 1) = SplitMerge.findRoot(parent, pairs(k, 1));
                    pairs(k, 2) = SplitMerge.findRoot(parent, pairs(k, 2));
                end

                pairs = sort(pairs, 2);
                pairs = pairs(pairs(:, 1) ~= pairs(:, 2), :);
                pairs = unique(pairs, 'rows');

                if isempty(pairs)
                    break;
                end

                means = sums ./ max(areas, 1);

                differences = abs(means(pairs(:, 1)) - means(pairs(:, 2)));

                % Select the most similar adjacent pair
                [minDifference, idx] = min(differences);
                if minDifference > threshold
                    break;
                end

                a = pairs(idx, 1);
                b = pairs(idx, 2);

                % Merge b into a
                parent(b) = a;

                areas(a) = areas(a) + areas(b);
                sums(a) = sums(a) + sums(b);

                areas(b) = 0;
                sums(b) = 0;
            end

            for k = 1:n
                parent(k) = SplitMerge.findRoot(parent, k);
            end

            labels = reshape(parent(double(labels(:))), size(labels));

            [~, ~, newLabels] = unique(labels(:));
            labels = reshape(newLabels, size(labels));
        end

        function pairs = getAdjacentPairs(labels)
            % Horizontal neighboring pixels
            A = labels(:, 1:end-1);
            B = labels(:, 2:end);

            horizontal = [double(A(:)), double(B(:))];

            % Vertical neighboring pixels
            A = labels(1:end-1, :);
            B = labels(2:end, :);

            vertical = [double(A(:)), double(B(:))];

            pairs = [horizontal; vertical];
            pairs = pairs(pairs(:, 1) ~= pairs(:, 2), :);
            pairs = unique(sort(pairs, 2), 'rows');
        end

        function root = findRoot(parent, node)
            % Find representative of a merged region
            root = node;

            while parent(root) ~= root
                root = parent(root);
            end
        end
    end
end
