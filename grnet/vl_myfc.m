function [Y, Y_w] = vl_myfc(X, W, dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer

[n1,n2,n3,n4] = size(X);

for ix = 1 : n3
    x_t = X(:,:,ix,:);
    X_t(:,ix) = x_t(:);
end
if nargin < 3
    Y = W'*X_t;
else
    Y = W * dzdy; % patial derivative with respect to x --- dz/dx
    Y_w = X_t * dzdy'; % dz/dy
end