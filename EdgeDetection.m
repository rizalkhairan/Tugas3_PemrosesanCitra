classdef EdgeDetection
    methods (Static)
        function mask = segment(edgeImage, radius)

            % Threshold edge to binary
            BW = EdgeDetection.toBinary(edgeImage);
        
            % Close gaps in boundaries
            SE = strel('disk', radius);
            BW = imclose(BW, SE);
        
            % Fill enclosed regions
            BW = imfill(BW, 'holes');
        
            % Select largest connected region
            CC = bwconncomp(BW, 8);
        
            mask = false(size(BW));
        
            if CC.NumObjects > 0
                areas = cellfun(@numel, CC.PixelIdxList);
                [~, idx] = max(areas);
                mask(CC.PixelIdxList{idx}) = true;
            end

            mask = repmat(uint8(mask) * 255, [1 1 3]);
        end

        function BW = laplace(I, neighbor)
            switch neighbor
                case 4
                    H = [0 1 0; 1 -4 1; 0 1 0];
                case 8
                    H = [1 1 1; 1 -8 1; 1 1 1];
                otherwise
                    error("Invalid neighbor count");
            end

            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            I = im2double(I);

            res = conv2(I, H, 'same');
            BW = im2uint8(mat2gray(abs(res)));
        end

        function BW = log(I, sigma)
            if sigma <= 0
                error("Sigma must be positive.");
            end

            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            I = im2double(I);

            radius = ceil(3 * sigma);
            [x, y] = meshgrid(-radius:radius, -radius:radius);

            % amazing
            H = ((x.^2 + y.^2 - 2*sigma^2) ./ (2*pi*sigma^6)) .* exp(-(x.^2 + y.^2) / (2*sigma^2));
            H = H-mean(H(:));

            res = conv2(I, H, 'same');
            BW = uint8(255 * mat2gray(abs(res)));
        end

        function BW = sobel(I)
            sx = [-1 0 1; -2 0 2; -1 0 1];
            sy = [1 2 1; 0 0 0; -1 -2 -1];
            
            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            I = im2double(I);

            gx = conv2(I, sx, 'same');
            gy = conv2(I, sy, 'same');
            BW = uint8(255 * mat2gray(hypot(gx, gy)));
        end

        function BW = prewitt(I)
            sx = [-1 0 1; -1 0 1; -1 0 1];
            sy = [1 1 1; 0 0 0; -1 -1 -1];
            
            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            I = im2double(I);

            gx = conv2(I, sx, 'same');
            gy = conv2(I, sy, 'same');
            BW = uint8(255 * mat2gray(hypot(gx, gy)));
        end

        function BW = roberts(I)
            r1 = [1 0; 0 -1];
            r2 = [0 1; -1 0];

            if size(I, 3) == 3
                I = rgb2gray(I);
            end
            I = im2double(I);
            gx = conv2(I, r1, 'same');
            gy = conv2(I, r2, 'same');
            BW = uint8(255 * mat2gray(abs(gx) + abs(gy)));
        end

        function BW = canny(I, sigma, lowThreshold, highThreshold)
            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            BW = edge(I, 'Canny', [lowThreshold highThreshold], sigma);
            BW = repmat(uint8(BW) * 255, [1 1 3]);
        end

    end

    methods (Static, Access = private)
        function BW = toBinary(edgeImage)
            if size(edgeImage, 3) == 3
                edgeImage = rgb2gray(edgeImage);
            end

            if islogical(edgeImage)
                BW = edgeImage;
            else
                I = mat2gray(edgeImage);
                threshold = graythresh(I);
                BW = I > threshold;
            end
        end
    end
end
