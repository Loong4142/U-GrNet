function [Y, parW, parB] = vl_mymarginloss(X, c, epoch, count1, doder, parW, parB, dzdy_recon, dzdx_log)

% metric learning term
[n1,n2,n3,n4] = size(X);
tau1 = -0.05; % a throshold to control the inter-class manifold margin
tau2 = 0.1; % a throshold to control the intra-class manifold margin

new_count = count1;
dzdy_l2 = single(1);

% eta = 0.1; %CG·
eta = 0.1; %FPHA
eta = 0.99^floor(epoch / 10) * eta;
% eta = 0.8^floor(epoch / 200) * eta;

Nw = zeros(1, n3, n4);
Nb = zeros(1, n3, n4);
Sw = zeros(1, n3, n4); % intra-class scatter
Sb = zeros(1, n3, n4); % inter-class scatter

temp_dev_Sw = zeros(n1,n2,n3,n4);
temp_dev_Sb = zeros(n1,n2,n3,n4);

use_parfor = cell(1,n3);
for i = 1 : size(use_parfor,2)
    use_parfor{i} = X;
end

if doder
    parfor j = 1 : n3
        K1 = 4; % the number of intra-manifold neighboring points
        num_eachclass = find(c==c(j));
        temp_X = use_parfor{j};

        for z = 1 : n4
            Xi = temp_X(:,:,j,z);
            Sw_temp = zeros(1,length(num_eachclass));
            temp_dev_Sw_store = zeros(n1,n2,length(num_eachclass));
            for k = 1 : length(num_eachclass)
                Nw(:,j,z) = Nw(:,j,z) + 1;
                Xj = temp_X(:,:,num_eachclass(k),z);
                proj_xi = Xi*Xi';
                proj_xj = Xj*Xj';
                temp = proj_xi - proj_xj;
                temp_dev_Sw_store(:,:,k) = 2*temp*Xi;
                Sw_temp(k) = norm(temp, 'fro') * norm(temp, 'fro');
            end
            [~,idx] = sort(Sw_temp);
            if (length(num_eachclass) < K1)
                K1 = length(num_eachclass);
            end

            dist_temp = Sw_temp(idx(:,1:K1)); % get the first K1 smallest distances
            Sw(:,j,z) = sum(dist_temp); % this is the scatter matrix composed by K1 nearest samples
            Sw(:,j,z) = 2^(-1/2) * Sw(:,j,z);
            temp_dev_Sw(:,:,j,z) = sum(temp_dev_Sw_store(:,:,idx(:,1:K1)),3);
            if K1 == 1
                Nw(:,j,z) = K1 +1;
            else
                Nw(:,j,z) = K1;
            end
        end
    end

    parfor j = 1:n3
        K2 = 3; % the number of the inter-manifold neighboring points
        num_difclass=find(c~=c(j)); %
        temp_X = use_parfor{j};

        for z = 1 : n4
            Xi = temp_X(:,:,j,z);
            Sb_temp = zeros(1,length(num_difclass));
            temp_dev_Sb_store = zeros(n1,n2,length(num_difclass));
            for k = 1:length(num_difclass)
                Xj = temp_X(:,:,num_difclass(k),z);
                Nb(:,j,z) = Nb(:,j,z) + 1;
                proj_xi = Xi*Xi';
                proj_xj = Xj*Xj';
                temp = proj_xi - proj_xj;
                temp_dev_Sb_store(:,:,k) = 2*temp*Xi;
                Sb_temp(k) = norm(temp, 'fro') * norm(temp, 'fro');
            end
            [~,idx] = sort(Sb_temp);
            if (length(num_difclass) < K2)
                K2 = length(num_difclass);
            end
            dist_temp = Sb_temp(idx(:,1:K2)); % get the first K2 smallest distances
            Sb(:,j,z) = sum(dist_temp); % this is the scatter matrix composed by K2 nearest samples
            Sb(:,j,z) = 2^(-1/2) * Sb(:,j,z);
            temp_dev_Sb(:,:,j,z) = sum(temp_dev_Sb_store(:,:,idx(:,1:K2)),3);
            Nb(:,j,z) = K2;
        end
    end
    parW.Sw = Sw;
    parW.Nw = Nw;
    parW.temp_dev_Sw = temp_dev_Sw;
    parB.Sb = Sb;
    parB.Nb = Nb;
    parB.temp_dev_Sb = temp_dev_Sb;
else
    Sw = parW.Sw;
    Nw = parW.Nw;
    temp_dev_Sw = parW.temp_dev_Sw;
    Sb = parB.Sb;
    Nb = parB.Nb;
    temp_dev_Sb = parB.temp_dev_Sb;
end

beta = 0.4;

temp_scatter = zeros(1,n3,n4);
Y = zeros(n1,n2,n3,n4);
Y_sum = 0;

for m = 1 : n3
    for n = 1 : n4
        Sw_each = Sw(:,m,n) / (Nw(:,m,n)-1);
        Sb_each = Sb(:,m,n) / Nb(:,m,n);
        d_inter = Sw_each - Sb_each;
        d_intra = Sw_each;
        % temp_scatter(m) = d_inter + alpha * d_intra;
        % Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if (d_inter <= tau1 && d_intra <= tau2)
            temp_scatter(:,m,n) = tau1 + beta * tau2;
            while temp_scatter(:,m,n)>= 700
                temp_scatter(:,m,n) = temp_scatter(:,m,n)/1.5;
            end
            Y_sum = Y_sum + log(1 + exp(temp_scatter(:,m,n)));
            if nargin < 6
                Y = eta * Y_sum;
            else
                Y(:,:,m,n) = 0 + dzdy_recon(:,:,m,n) + dzdx_log(:,:,m,n); % 0 --> (3).1) of Section III-E
            end
        elseif d_inter <= tau1 && d_intra > tau2
            temp_scatter(:,m,n) = tau1 + beta * d_intra;
            while temp_scatter(:,m,n)>= 700
                temp_scatter(:,m,n) = temp_scatter(:,m,n)/1.5;
            end
            Y_sum = Y_sum + log(1 + exp(temp_scatter(:,m,n)));
            if nargin < 6
                Y = eta * Y_sum;
            else
                dev_part1 = 1 / (1 + exp(temp_scatter(:,m,n)));
                dev_part2 = exp(temp_scatter(:,m,n));
                temp_dev_Sw_each = temp_dev_Sw(:,:,m,n) / (Nw(:,m,n) - 1);
                dev_part3 = 2 * beta * temp_dev_Sw_each;
                dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(n1,n2), dzdy_l2));
                Y(:,:,m,n) = eta * dev_l2 + dzdy_recon(:,:,m,n) + dzdx_log(:,:,m,n);  % dev_l2 --> (3).2) of Section III-E
            end
        elseif d_inter > tau1 && d_intra <= tau2
            temp_scatter(:,m,n)= d_inter + beta * tau2;
            while temp_scatter(:,m,n)>= 700
                temp_scatter(:,m,n) = temp_scatter(:,m,n)/1.5;
            end
            Y_sum = Y_sum + log(1 + exp(temp_scatter(:,m,n)));
            if nargin < 6
                Y = eta * Y_sum;
            else
                dev_part1 = 1 / (1 + exp(temp_scatter(:,m,n)));
                dev_part2 = exp(temp_scatter(:,m,n));
                temp_dev_Sw_each = temp_dev_Sw(:,:,m,n) / (Nw(:,m,n) - 1);
                temp_dev_Sb_each = temp_dev_Sb(:,:,m,n) / Nb(:,m,n);
                dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each;
                dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(n1,n2), dzdy_l2));
                Y(:,:,m,n) = eta * dev_l2 + dzdy_recon(:,:,m,n) + dzdx_log(:,:,m,n);  % dev_l2 --> (3).3) of Section III-E
            end
        else
            temp_scatter(:,m,n) = d_inter + beta * d_intra;
            if temp_scatter(:,m,n)>= 700
                temp_scatter(:,m,n) = temp_scatter(:,m,n)/1.5;
            end
            Y_sum = Y_sum + log(1 + exp(temp_scatter(:,m,n)));
            if nargin < 6
                Y = eta * Y_sum;
            else
                dev_part1 = 1 / (1 + exp(temp_scatter(:,m,n)));
                dev_part2 = exp(temp_scatter(:,m,n));
                temp_dev_Sw_each = temp_dev_Sw(:,:,m,n) / (Nw(:,m,n) - 1);
                temp_dev_Sb_each = temp_dev_Sb(:,:,m,n) / Nb(:,m,n);
                dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each + 2 * beta * temp_dev_Sw_each;
                dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(n1,n2), dzdy_l2));
                Y(:,:,m,n) = eta * dev_l2 + dzdy_recon(:,:,m,n) + dzdx_log(:,:,m,n);  % dev_l2 --> (3).4) of Section III-E
            end
        end
    end
end



