function [p,p0]=robust_regression(D,lambda)
% robust regression - estimates displacement parameters through outlier
% pruning

D0=D;
S=zeros(size(D));
p0=nanmedian(D0-nanmedian(D0,2),1);

for t=1:10
    p=nanmedian(D-l1tf(nanmedian(D,2),lambda),1);
    P=D0-l1tf(nanmedian(D,2),lambda);
    S=abs(zscore(P,[],1))>3;S=or(S,S');
    D=D0;
    D(S==1)=nan;
end
end