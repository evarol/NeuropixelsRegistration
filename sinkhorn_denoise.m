function [X,obj]=sinkhorn_denoise(X,q,iter)
if nargin<3
    iter=20;
end
for t=1:iter
    if nargout>1
        Xold=X;
    end
    X=max(X-quantile(X,q,1),0);
    X=max(X-quantile(X,q,2),0);
    if nargout>1
        obj(t)=norm(X-Xold,'fro');
    end
end
end

