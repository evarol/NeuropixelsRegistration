function p=psolver_weighted(Dy,C,reg_fit_sigma,time_sigma,robust_sigma)
if nargin<5
    robust=0;
else
    robust=1;
end


[I,J,~]=find(~isnan(Dy));
S=~isnan(Dy);
V=Dy(S==1);
W1=exp(-C(S==1)/reg_fit_sigma); %Weights based on registration fit
W2=exp(-squareform(pdist((1:size(Dy,1))'))/time_sigma);W2=W2(S==1);%Weights based on time similarity
W=W2.*W1; % Combined weights
M=sparse((1:size(I,1))',I,ones(size(I)));
N=sparse((1:size(I,1))',J,ones(size(I)));
A=M-N;


%% non-robust regression

if robust==0
    p=lsqr(A.*W,V.*W);obj=[]; %weighted regression
    
else
    %% robust regression
    idx=(1:size(A,1))';
    pold=nan(size(Dy,2),1);
    for t=1:20
        p=lsqr(A(idx,:).*W(idx,:),V(idx,:).*W(idx,:));  %weighted regression
        pold=p;
        idx=find(abs(zscore(A*p-V))<=robust_sigma);
    end
end
end