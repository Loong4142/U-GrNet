function [Y,R] = vl_myreorth(i, R, dzdy, dev_sc)
%re-orthonormalization (ReOrth) layer

X = R.x;

[n1,n2,n3,n4] = size(X);


Y = zeros(n1,n2,n3,n4);

if isempty(R.aux)==1
    Qs = zeros(n1,n2,n3,n4);
    Rs = zeros(n2,n2,n3,n4);
    for ix = 1  : n3
        if n4 == 1
            [Qs(:,:,ix),Rs(:,:,ix)] = qr(X(:,:,ix),0);
            Y(:,:,ix) = Qs(:,:,ix);
        else
            for iy = 1 : n4
                [Qs(:,:,ix,iy),Rs(:,:,ix,iy)] = qr(X(:,:,ix,iy),0);

                Y(:,:,ix,iy) = Qs(:,:,ix,iy);
            end
        end
    end
    R.aux{1} = Qs;
    R.aux{2} = Rs;
else
    Qs = R.aux{1};
    Rs = R.aux{2};
    for ix = 1  : n3
        if n4 == 1
            Q = Qs(:,:,ix); R = Rs(:,:,ix);
            T = dzdy(:,:,ix);
            dzdx = Compute_Gradient_QR(Q,R,T);
            if i == 23
                Y(:,:,ix) =  dzdx;
            else
                Y(:,:,ix) =  dzdx + dev_sc(:,:,ix);
            end

        else
            for iy = 1 : n4
                Q = Qs(:,:,ix,iy); R = Rs(:,:,ix,iy);
                T = dzdy(:,:,ix,iy);
                dzdx = Compute_Gradient_QR(Q,R,T);
                if i == 23
                    Y(:,:,ix,iy) =  dzdx;
                else
                    Y(:,:,ix,iy) =  dzdx+ dev_sc(:,:,ix,iy);
                end
            end
        end
    end
end


function dzdx = Compute_Gradient_QR(Q,R,T)
m = size(Q,1);
dLdC = double(T);
dLdQ = dLdC;

S = eye(m)-Q*Q';
dzdx_t0 = Q'*dLdQ;
dzdx_t1 = tril(dzdx_t0,-1);
dzdx_t2 = tril(dzdx_t0',-1);
% 使用 QR 分解求解线性系统
[~, R_inv] = qr(R);

% 计算梯度 dzdx
dzdx = (S'*dLdC + Q*(dzdx_t1 - dzdx_t2)) * R_inv';
