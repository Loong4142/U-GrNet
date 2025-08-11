function Y = vl_myreconstructionloss(X, X_ori, epoch, dzdy)
% this function is designed to implement the decode term with the reconstruction function
% Date:
% Author:
% Coryright£º
% Note!!!!!:to make the code currectly run, I adjust the line of 96-107 of steifelfactory.m
[n1,n2,n3,n4] = size(X);

dzdy = single(zeros(n1,n2,n3,n4));

dzdy_l3 = single(1);
% gamma = 0.01;% CG
gamma = 0.01; % FPHA
count = epoch;
%gamma = 0.8^floor(epoch / 20) * gamma;
dist_sum = zeros(1,n3); % save each pair dist
Y = zeros(n1,n2,n3,n4); % save obj or dev
dev_term = zeros(n1,n2,n3,n4); % save each pair' derivation

for i = 1 : n3
    for w = 1: n4
        if n4 == 1
            temp = X(:,:,i) - X_ori(:,:,i); %
            dev_term(:,:,i) = 2 * temp;  %
            dist_sum(i) = dist_sum(i) + norm(temp,'fro') * norm(temp,'fro'); %
        else
            temp = X(:,:,i,w) - X_ori(:,:,i); %
            dev_term(:,:,i,w) = 2 * temp;  %
            dist_sum(i) = dist_sum(i) + norm(temp,'fro') * norm(temp,'fro'); %
        end
    end
end

if nargin < 4
    Y = gamma * (sum(dist_sum) / (n3*n4)); % the obj of this loss function
else
    for j = 1 : n3
        for k  = 1 : n4
            if n4 == 1
                dev_l3 = bsxfun(@times, dev_term(:,:,j), bsxfun(@times, ones(n1,n2), dzdy_l3));
                Y(:,:,j) = gamma * dev_l3 + dzdy(:,:,j); % the sum of reconstruction term and softmax term
            else
                dev_l3 = bsxfun(@times, dev_term(:,:,i,k), bsxfun(@times, ones(n1,n2), dzdy_l3));
                Y(:,:,i,k) = gamma * dev_l3 + dzdy(:,:,i,k); % the sum of reconstruction term and softmax term
            end
        end
    end
end
end

