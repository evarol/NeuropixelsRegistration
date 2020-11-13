function loc=spatial_icdf(A,q)
loc=zeros(length(q),size(A,2));
for i=1:length(q)
[~,idx]=max(flipud(A<=q(i)),[],1);idx=size(A,1)-idx+1;
loc(i,:)=idx;
end
end