function [res] = vl_myforbackward(net, x, dzdy, res, epoch, count1, varargin)
% vl_myforbackward  evaluates a simple SPDNet

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-5; % this parameter is worked in the ReEig Layer
opts.p = 10;

n = numel(net.layers) ; % count the number of layers

if (nargin <= 2) || isempty(dzdy)
    doder = false ;
else
    doder = true ; % this variable is used to control when to compute the derivative
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
else
    cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ... % this gradient is necessary for computing the gradients in the layers below and updating their parameters
        'dzdw', cell(1,n+1), ... % this gradient is required for updating W
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
    res(1).x = x ;
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
    if opts.skipForward
        break;
    end
    l = net.layers{i} ; % each net layer stores two components: (1) layer type (2) weight
    res(i).time = tic ; % count the time spend on each layer
    if isnan(res(i).x(1))
        disp(l.type)
    end
    switch l.type
        case 'frmap'
            res(i+1).x = vl_myfrmap(res(i).x, l.weight) ;
        case 'fc'
            res(i+1).x = vl_myfc(res(i).x, l.weight) ;
        case 'reorth'
            [res(i+1).x, res(i)] = vl_myreorth(i, res(i)) ;
        case 'add'
            sc = res(i-1).x;
            res(i+1).x = vl_myadd(res(i).x,sc);
        case 'marginloss'
            if doder
                [res(i+1).obj, WW, BB] = vl_mymarginloss(res(i).x, l.class, epoch, count1, doder) ;  % this is the metric learning regularizer
                res(i+1).ww = WW;
                res(i+1).bb = BB;
            else
                res(i+1).obj = 0;
            end
            res(i+1).x = res(i).x;

            % res(i+1).obj = 0.0;
            % res(i+1).x = res(i).x;
        case 'reconstructionloss'
            res(i+1).obj = vl_myreconstructionloss(res(i).x, res(1).x, epoch); %
            res(i+1).x = res(8).x;
        case 'softmaxloss'
            res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ;
        case 'projmap'
            if i == 11
                sc = res(i-6).x;
            elseif i == 18
                sc = res(i-15).x;
            else
                sc = res(i).x;
            end
            res(i+1).x = vl_myprojmap(sc) ;
        case 'orthmap'
            [res(i+1).x, res(i)] = vl_myorthmap(res(i),opts.p) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end
    % optionally forget intermediate results
    forget = opts.conserveMemory ;
    forget = forget & (~doder || strcmp(l.type, 'relu')) ;
    forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
    if forget
        res(i).x = [] ;
    end
    if gpuMode & opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice) ;
    end
    res(i).time = toc(res(i).time) ;
end