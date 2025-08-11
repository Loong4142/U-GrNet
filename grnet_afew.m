function [net, info] = grnet_afew(varargin)
%set up the path

confPath; % upload the path of some toolkits to the workspace
%parameter setting

opts.dataDir = fullfile('./data/CG') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'sample_for_GrNet.mat'); % get the data
opts.batchSize = 20; % original is 30 
opts.test.batchSize = 1;
opts.numEpochs = 4000; % maximum number of epoches is 500, this can be adjusted
opts.gpus = [] ;   
opts.learningRate = 0.01 * ones(1,opts.numEpochs); % original lr is 0.01 
opts.weightDecay = 0.0005 ; 
opts.continue = 1;
%spdnet initialization
net = grnet_init_afew() ; % 
%loading metadata 
load(opts.imdbPathtrain) ;
%spdnet training
[net, info] = grnet_train_afew(net, gr_train, opts);