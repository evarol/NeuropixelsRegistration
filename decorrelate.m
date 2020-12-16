function X=decorrelate(X,iter)
if nargin<2
    iter=10;
end

for i=1:iter
    X=zscore(X,[],2);
    X=zscore(X,[],1);
end


end