function [depths,amps,times,widths,pz0]=simulated_localizations(num_bins)

vec=@(x)(x(:));
%% Simulated data generation

z_box=[0 3800];
x_box=[0 60];
ptp_box=[5 20];
num_obj=50;
spread=200;
ptp_variance=5;
% num_bins=100;
movement=1;
for k=1:num_obj
    mu(k,:)=[randi([x_box(1) x_box(2)]) randi([z_box(1) z_box(2)]) randi([ptp_box(1) ptp_box(2)])];
    sigma(:,:,k)=zeros(3);
    sigma(1:2,1:2,k)=rand(1)*spread*eye(2);
    sigma(3,3,k)=rand(1)*ptp_variance;
    p(k)=rand(1);
end


%% establish random walk movement
pz0(1)=0;
for t=2:num_bins
    pz0(t)=pz0(t-1)+movement*randn(1);
end

%% translate localizations
for t=1:num_bins
    gm = gmdistribution(mu+[0 pz0(t) 0],sigma,p);
    X{t} = random(gm,1000);
    [~,idx]=sort(X{t}(:,2));
    X{t}=X{t}(idx,:);
%     toremove=find(or(or(X{t}(:,1)>x_box(2),X{t}(:,1)<x_box(1)),or(X{t}(:,2)>z_box(2),X{t}(:,2)<z_box(1))));
%     X{t}(toremove,:)=[];
end

depths=[];widths=[];times=[];amps=[];
for t=1:num_bins
    depths=[depths;X{t}(:,2)];
    widths=[widths;X{t}(:,1)];
    amps=[amps;X{t}(:,3)];
    times=[times;repmat(t,size(X{t},1),1)];
end