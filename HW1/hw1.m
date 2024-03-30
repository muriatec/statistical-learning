load("C:\Users\Muriate_C\Desktop\ECE 271A\HW1\TrainingSamplesDCT_8.mat")

% Compute priors
prob_bg = size(TrainsampleDCT_BG, 1) / (size(TrainsampleDCT_BG, 1) + size(TrainsampleDCT_FG, 1))
prob_fg = size(TrainsampleDCT_FG, 1) / (size(TrainsampleDCT_BG, 1) + size(TrainsampleDCT_FG, 1))
%%
bg_train_indices = find2ndLargest(TrainsampleDCT_BG)
fg_train_indices = find2ndLargest(TrainsampleDCT_FG)

h_bg = histogram(bg_train_indices, 64, "BinEdges", [1:64], 'Normalization','probability')
h_fg = histogram(fg_train_indices, 64, "BinEdges", [1:64], 'Normalization','probability')
%%
% Compute likelihood
prob_x_bg = histcounts(bg_train_indices, [1:65])
prob_x_bg = prob_x_bg / sum(prob_x_bg)
prob_x_fg = histcounts(fg_train_indices, [1:65])
prob_x_fg = prob_x_fg / sum(prob_x_fg)
%%
% ZigZagPattern = ZigZagPattern - 1
% ZigZagPattern = int8(ZigZagPattern)
%%
img = imread("C:\Users\Muriate_C\Desktop\ECE 271A\HW1\cheetah.bmp")
img = im2double(img)

% Zero Padding
% right = zeros(255, 7);
% bottom = zeros(7, 277);
% img_pad = [[img right]; bottom]
left = zeros(255, 3);
right = zeros(255, 4);
up = zeros(3, 277);
bottom = zeros(4, 277);
img_pad = [up; [left img right]; bottom]

% DCT
img_dct = dct_8(img, img_pad);
img_dct = abs(img_dct)

% ZigZag Scan
img_scan = blockproc(img_dct, [8 8], @(block_struct) ZigZagScan(block_struct.data, ZigZagPattern))
features = blockproc(img_scan, [1, 64], @(block_struct) find2ndLargest(block_struct.data));
features = int8(features)

% Create binary mask using BDR
mask = blockproc(features, [1, 1], @(block_struct) BDR(block_struct.data, prob_x_bg, prob_x_fg, prob_bg, prob_fg));
mask = int8(mask)
imagesc(mask)
colormap(gray(255))
%%
ground_truth = imread("C:\Users\Muriate_C\Desktop\ECE 271A\HW1\cheetah_mask.bmp")
ground_truth = im2double(ground_truth)
imagesc(ground_truth)
colormap(gray(255))

ground_truth = int8(ground_truth)

% Compute probability of errors
% Method 1
p_error = 0;
diff = ground_truth - mask
diff_feature = features .* diff
false_fg = int8.empty;
false_bg = int8.empty;
x = (1:64);
for i = 1:size(diff_feature, 1)
    for j = 1:size(diff_feature, 2)
        if diff_feature(i, j) < 0 % Find false foreground (ground truth:0, predicted Y:1)
            false_fg = [false_fg -diff_feature(i, j)];
        end
        if diff_feature(i, j) > 0 % Find false background (ground truth:1, predicted Y:0)
            false_bg = [false_bg diff_feature(i, j)];
        end
    end
end
false_fg_x = intersect(x, false_fg)
false_bg_x = intersect(x, false_bg)
for i = 1:length(x)
    p_error = p_error + prob_x_bg(1, i) * prob_bg * ismember(i, false_fg_x) ...,
            + prob_x_fg(1, i) * prob_fg * ismember(i, false_bg_x);
end
p_error

% Method 2
p_error_1 = 0;
for i = 1:64
    p_error_1 = p_error_1 + min(prob_x_bg(1, i) * prob_bg, prob_x_fg(1, i) * prob_fg);
end
p_error_1

% Method 3
p_fg_gt = sum(sum(ground_truth==1)) / (size(ground_truth, 1) * size(ground_truth, 2));
p_bg_gt = sum(sum(ground_truth==0)) / (size(ground_truth, 1) * size(ground_truth, 2));
p_error_2 = sum(sum(diff==1)) / sum(sum(ground_truth==1)) * p_fg_gt + sum(sum(diff==-1)) / sum(sum(ground_truth==0)) * p_bg_gt
p_error_3 = 1 - sum(sum(mask==ground_truth)) / (size(ground_truth, 1) * size(ground_truth, 2))
%%
function vector = ZigZagScan(matrix, pattern)
    vector = zeros(1, size(matrix, 1) * size(matrix, 2));
    for i = 1:size(matrix, 1)
        for j = 1:size(matrix, 2)
            position = pattern(i, j);
            vector(1, position) = matrix(i, j);
        end
    end
end

function indices = find2ndLargest(sample)
    indices = zeros(1, size(sample, 1));
    for i = 1:size(sample, 1)
        [~, index] = maxk(sample(i, :), 2);
        indices(1, i) = indices(1, i) + index(1, 2);
    end
end

function mask = BDR(feature, P_x_bg, P_x_fg, P_bg, P_fg)
    if P_x_bg(1, feature) * P_bg >= P_x_fg(1, feature) * P_fg
        mask = 0;
    else
        mask = 1;
    end
end

function dct = dct_8(img, img_pad)
    dct = zeros(size(img, 1) * 8, size(img, 2) * 8);
    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
            dct((8*i-7):(8*i), (8*j-7):(8*j)) = dct2(img_pad(i:i+7, j:j+7));
        end
    end
end