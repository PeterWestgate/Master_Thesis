% BUILD GIST-BASED RDM
%
% Consider only the control images, extract GIST features for each 
% image, and build a distance metric-based RDM of size 80x80. Serve
% as a theoretical model to feed into the inverse MDS RSA.
%
% GIST will build a 512-long vector of low-level visual features per 
% image.


%% Directories

% Source and toolbox path
gistToolboxPath = '/Users/tim/Desktop/fmri_analysis/RSA/src/gistdescriptor';
inverseMDSpath = '/Users/tim/Desktop/inverse_mds';

% Output path
outDir = fullfile(inverseMDSpath, 'models');
outFilename = 'gist80_model.tsv';
outPath = fullfile(outDir, outFilename);

% Add source and the GIST toolbox to the path
addpath(srcPath);
addpath(gistToolboxPath);

% Image path
imgPath = '/Users/tim/github_repos/fmri-categorisation-task/src/stimuli';

%% Image and conditions

% Define categories and manipulations in their correct order
manipulations = {'control', 'clutter', 'deletion', 'occlusion', 'scrambling'};
categories = {'person', 'cat', 'bird', 'banana', 'firehydrant', 'tree', 'bus', 'building'};

% Generate condition list in the desired order
conditions = strings(length(manipulations)*length(categories), 1);
idx = 1;
for c = 1:length(categories)
    for m = 1:length(manipulations)
        conditions(idx) = strcat(manipulations{m}, '_', categories{c});
        idx = idx + 1;
    end
end

%% Parameters

% Define GIST model parameters
num_conditions = length(conditions);
param.orientationsPerScale = [8 8 8 8]; % Number of orientations per scale
param.numberBlocks = 4; % Grid size
param.fc_prefilt = 4; % Pre-filtering constant

% Decide on the distance metric to use
distance_metric = 'correlation';

% Define a category order
category_order = ['person', 'cat', 'bird', 'banana', 'firehydrant', 'tree', 'bus', 'building'];

%% Load images 

% List all the images
image_filenames = dir([imgPath, '/*.png']);

% Only keep the control images
image_filenames = image_filenames(contains({image_filenames.name}, 'control'));

% Extract the number of images per condition
num_images = length(image_filenames);

%% Compute GIST descriptors

% Start an object to store the gist features per image
gist_features = zeros(num_images, 512); % Assuming GIST outputs 512 features

% Extract gist features for each image - with a loading bar
h = waitbar(0, 'Processing images...');
for i = 1:num_images
    img = imread(fullfile(image_filenames(i).folder, image_filenames(i).name));
    gist_features(i, :) = LMgist(img, '', param);
    
    % Update waitbar
    waitbar(i / num_images, h, sprintf('Processing %d/%d images...', i, num_images));
end
close(h);

%% Pairwise GIST features distance matrix

% Compute pairwise Euclidean distances (80x80 matrix)
gist_distances = pdist2(gist_features, gist_features, distance_metric);


%% Create an image name array

img_names = strings(num_images, 1); % Preallocate for image names
for i = 1:num_images
    img_names(i) = string(strrep(image_filenames(i).name, '.png', ''));
end

%% Save as .tsv

% Turn the matrix into an explicit table with labels
distance_table = array2table(gist_distances, ...
    'RowNames', img_names, 'VariableNames', img_names);

% Save table as TSV (tab-delimited text)
writetable(distance_table, outPath, 'FileType', 'text', 'Delimiter', '\t', 'WriteRowNames', true);
