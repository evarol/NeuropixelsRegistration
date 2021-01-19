function [p,p0]=robust_regression(D,lambda)
% robust regression - estimates displacement parameters through outlier
% pruning

D0=D;
S=zeros(size(D));
p0=nanmean(D0-nanmean(D0,2),1);

for t=1:10
    p=nanmean(D-l1tf(nanmean(D,2),lambda),1);
    P=D0-l1tf(nanmean(D,2),lambda);
    S=abs(zscore(P,[],1))>2;S=or(S,S');
    D=D0;
    D(S==1)=nan;
end
end