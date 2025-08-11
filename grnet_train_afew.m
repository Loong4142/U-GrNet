function [net, info] = spdnet_train_afew(net, gr_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(gr_train.gr.set==1) ; % 1 represents the training samples
opts.val = find(gr_train.gr.set==2) ; % 2 indicates the testing samples
count = 0;
%load my_label % the label of training data
%load data_indx_2

for epoch = 3999 : 4000

    learningRate = opts.learningRate(epoch); 
    %% fast-forward to last checkpoint
    modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
    if opts.continue
        if exist(modelPath(epoch),'file')
            if epoch == opts.numEpochs
                load(modelPath(epoch), 'net', 'info') ;
            end
            continue ;
        end
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1), 'net', 'info') ;
        end
    end

    train = opts.train(randperm(length(opts.train))) ; % data_label; shuffle, to make the training data feed into the net in disorder
    val = opts.val; % the train data is in order to pass the net
    [net,stats.val,acc] = process_epoch(opts, epoch, gr_train, val, 0, net) ;
    fprintf('The accuracy of the classification results is %.4f %%',acc);
end


function [net,stats,acc] = process_epoch(opts, epoch, gr_train, trainInd, learningRate, net)

training = learningRate > 0 ;
count1 = 0;

if training
    mode = 'training' ;
else
    mode = 'validation' ;
end

stats = [0 ; 0; 0; 0; 0] ;
% stats = [0 ; 0; 0] ; % for softmax

numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;
flag = 0;
grPath = [gr_train.grDir '\' gr_train.gr.name{trainInd(1)}];
load(grPath);[n1,n2] = size(Y1);

for ib = 1 : batchSize : length(trainInd) % select the training samples. Here, 10 pairs of samples per group
    flag = flag + 1;
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ; %
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;  %
    else
        batchSize_r = batchSize;
    end
    gr_data = zeros(n1,n2,batchSize_r); % store the data in each batch
    gr_label = zeros(batchSize_r,1); % store the label of the data in each batch
    for ib_r = 1 : batchSize_r

        grPath = [gr_train.grDir '\' gr_train.gr.name{trainInd(ib+ib_r-1)}];
        load(grPath);

        gr_data(:,:,ib_r) = Y1;
        gr_label(ib_r) = gr_train.gr.label(trainInd(ib+ib_r-1));

    end
    net.layers{end}.class = gr_label; % one-hot vector is used to justify your algorithm generate a right label or not
    net.layers{7}.class = gr_label; %
    %forward/backward spdnet
    if training
        dzdy = one;
    else
        dzdy = [] ;
    end
    res = vl_myforbackward(net, gr_data, dzdy, res, epoch, count1) ; % its for-and-back-ward process

    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', gr_label)) ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    numDone = numDone + batchSize_r ;

    stats = stats+[batchTime ; res(end).x; res(8).obj; res(25).obj; error]; % works even when stats=[] res(15).obj
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' l1-sm: %.5f', stats(2)/numDone) ; % the value of objective function
    fprintf(' l2-ml: %.5f', stats(3)/numDone) ; % 0.0 in this net
    fprintf(' l2-rebud: %.5f', stats(4)/numDone) ;
    fprintf(' l-mix: %.5f', (stats(2) + stats(3) + stats(4))/numDone) ;
    fprintf(' error: %.5f', stats(5)/numDone) ;
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf(' lr: %.6f',learningRate);
    fprintf('\n') ;

end

acc = (1 - stats(5)/numDone)*100;