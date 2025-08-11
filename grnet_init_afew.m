function net = spdnet_init_afew_deep_v1(varargin)
% spdnet_init initializes the spdnet

rng('default');
rng(0) ;

opts.layernum = 6;

Winit = cell(opts.layernum,1);
opts.datadim = [100,80,60,40,60,80,100];       %CG

opts.skedim = [8, 8, 8, 8, 8, 8, 8];
for iw = 1 : opts.layernum % designed to initialize each cov kernel
    for i_s = 1 : opts.skedim(iw)
        if iw < 4
            A = rand(opts.datadim(iw));
            [U1, ~, ~] = svd(A * A');
            Winit{iw}.w(:,:,i_s) = U1(:,1:opts.datadim(iw+1))'; % the initialized filters are all satisfy column orthogonality, residing on the Stiefel manifolds
        else
            A = rand(opts.datadim(iw+1));
            [U1, ~, ~] = svd(A * A');
            Winit{iw}.w(:,:,i_s) =U1(:,1:opts.datadim(iw));
        end
    end
end

f=1/100 ;
classNum = 9;
fdim = size(Winit{iw-3}.w,1)*size(Winit{iw-3}.w,1)*opts.skedim(end);
theta = f * randn(fdim, classNum, 'single'); 
Winit{end+1}.w = theta; % the to-be-learned projection matrix of the FC layer

net.layers = {} ; % use to construct each layer of the proposed SPDNet
net.layers{end+1} = struct('type', 'frmap',...
                          'weight', Winit{1}.w) ;
net.layers{end+1} = struct('type', 'reorth') ; 
net.layers{end+1} = struct('type', 'frmap',...
                          'weight', Winit{2}.w) ;
net.layers{end+1} = struct('type', 'reorth') ;
net.layers{end+1} = struct('type', 'frmap',...
                          'weight', Winit{3}.w) ;
net.layers{end+1} = struct('type', 'reorth') ;

net.layers{end+1} = struct('type', 'marginloss') ; % is not used in our U-SPDNet

net.layers{end+1} = struct('type', 'frmap',...
                          'weight', Winit{4}.w) ;
net.layers{end+1} = struct('type', 'reorth') ;

net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'add') ; % lines 46-49, the first LFE module used for feature fusion
% net.layers{end+1} = struct('type', 'exp') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'orthmap') ;

net.layers{end+1} = struct('type', 'frmap',...
                          'weight', Winit{5}.w) ;
net.layers{end+1} = struct('type', 'reorth') ; 

net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'add') ; % lines 46-49, the first LFE module used for feature fusion
% net.layers{end+1} = struct('type', 'exp') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'orthmap') ;

net.layers{end+1} = struct('type', 'frmap',...
                          'weight', Winit{6}.w) ;
net.layers{end+1} = struct('type', 'reorth') ;

net.layers{end+1} = struct('type', 'reconstructionloss') ;

net.layers{end+1} = struct('type', 'projmap') ;

%% the following in the new framework may be removed 
net.layers{end+1} = struct('type', 'fc', ...
                           'weight', Winit{end}.w) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;
