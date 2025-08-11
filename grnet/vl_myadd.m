function Y = vl_myadd(X, X_sc, dzdy)
% VL_MYADD: defiend to implement the skip connection

[n1,n2,n3,n4] = size(X);
Y = zeros(n1,n1,n3,n4);

for ix = 1 : n3
    for iw = 1 : n4
        if n4 == 1
            temp = X(:,:,ix)+ X_sc(:,:,ix);
            Y(:,:,ix) = (1/2) * temp;
        else
            for iy = 1: n4
                temp = X(:,:,ix,iy)+ X_sc(:,:,ix,iy);
                Y(:,:,ix,iy) = (1/2) * temp;
            end
        end
    end
end
if nargin == 3
    for ix = 1 : n3
        if n4 == 1
            Y(:,:,ix) = (1/2) * dzdy(:,:,ix);
        else
            for iy = 1: n4
                Y(:,:,ix,iy) = (1/2) * dzdy(:,:,ix,iy);
            end
        end
    end
end
end

