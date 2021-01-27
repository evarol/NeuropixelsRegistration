clear all
clc
close all


globalTic=tic;
%% parameters
time_resolution  = 1; % in seconds
depth_resolution = 1; % in microns
subblocks=10; % number of rigid blocks of motion
blocksize=1000; % in microns
resolution=100; %the subpixel resolution i.e. resolution = 100 --> 1/100th of pixel resolution
robust_lambda=10000; %trend filtering penalty to smoothen displacement estimates

%% some helper functions
minmax = @(x)((x-min(x(:)))./max(x(:)-min(x(:))));


%% load data
addpath(genpath('/Users/erdem/Documents/Github/npy-matlab'));

amps = readNPY('/Users/erdem/Downloads/CSHL047_2020-01-20_001_alf_probe00/spikes.amps.npy');
depths = readNPY('/Users/erdem/Downloads/CSHL047_2020-01-20_001_alf_probe00/spikes.depths.npy');
times = readNPY('/Users/erdem/Downloads/CSHL047_2020-01-20_001_alf_probe00/spikes.times.npy');


%% allocate bin sizes
T=floor(min(times)):time_resolution:ceil(max(times));
Ybins=floor(min(depths)):depth_resolution:ceil(max(depths));

%% generate image representations
tic;
for t=1:length(T)-1
    data{t}(:,1)=depths(and(times>=T(t),times<=T(t+1)));
    data{t}(:,2)=amps(and(times>=T(t),times<=T(t+1)));
    for y=1:length(Ybins)-1
        I{t}(y,1)=mean(data{t}(and(data{t}(:,1)>=Ybins(y),data{t}(:,1)<=Ybins(y+1)),2));
    end
    
    clc
    fprintf(['Generating image representations (' num2str(t) '/' num2str(length(T)-1) ')...\n']);
    fprintf(['\n' repmat('.',1,50) '\n\n'])
    for tt=1:round(t*50/(length(T)-1))
        fprintf('\b|\n');
    end
    TT=toc;
    disp(['Time elapsed (minutes): ' num2str(TT/60) ' Time remaining (minutes): ' num2str(((length(T)-1)-t)*(TT/t)*(1/60))]);
    
end


%% generate a raster plot + correct for nans
for t=1:length(I)
    I{t}(isnan(I{t}))=0; % make depth levels with zero hits into zeros instead of nans
    X(:,t)=I{t}; % create a raster diagram
end

%% main decentralized registration routine
if subblocks==1
    lower_end=1;
    upper_end=size(I{1},1);
else
lower_end=linspace(1,size(I{1},1)-blocksize,subblocks);
upper_end=linspace(blocksize,size(I{1},1),subblocks);
end
for s=1:subblocks
    blockcoor{s}=floor(lower_end(s)):ceil(upper_end(s));
    for t=1:length(I)
        Is{t,s}=I{t}(blockcoor{s});
    end
end

subsampling_rate=log(length(I))/length(I); %% log(length(I))/length(I) is the subsampling rate - if the estimation is poor increase this


for s=1:subblocks
    [Dx{s},Dy{s}]=subsampled_pairwise_registration(Is(:,s),subsampling_rate,resolution); 
end


%% robustifying the displacement estimates

disp(['Centralizing the decentralized estimates...']);
for s=1:subblocks
[py{s},py0{s}]=robust_regression(Dy{s}',robust_lambda); %% if you get NaN's change robust_lambda to be higher or lower.
[px{s},px0{s}]=robust_regression(Dx{s}',robust_lambda);
end
disp(['Centralizing the decentralized estimates...(Done)']);


%% vector fields
V=nan(size(X,1),size(X,2),subblocks);
for s=1:subblocks
    V(blockcoor{s},:,s)=repmat(l1tf(py0{s}',1000)',[length(blockcoor{s}) 1]);
end

%% Undoing the translation with a  median vector field

Vf=nanmedian(V,3);

tic
for t=1:length(I)
    Xr(:,t)=interp1((1:size(X,1))',X(:,t),(1:size(X,1))'+Vf(:,t));
    clc
    fprintf(['Translating data (' num2str(t) '/' num2str(length(T)-1) ')...\n']);
    fprintf(['\n' repmat('.',1,50) '\n\n'])
    for tt=1:round(t*50/(length(T)-1))
        fprintf('\b|\n');
    end
    TT=toc;
    disp(['Time elapsed (minutes): ' num2str(TT/60) ' Time remaining (minutes): ' num2str(((length(T)-1)-t)*(TT/t)*(1/60))]);
end


%% visuals
figure(1)
subplot(5,1,1);
for s=1:subblocks
    hold on
    plot(py{s},'LineWidth',1);title('Decentralized displacement estimate + Unregistered data');xlabel('time bins');ylabel('displacement');grid on;set(gca,'xlim',[1 length(py{s})]);
end
subplot(5,1,[2 5]);imagesc(log1p(flipud(minmax(X))));colormap(flipud(gray(256)));title('mean ptp vs. time');xlabel('time bins');ylabel('depth');

figure(2)
subplot(5,1,1);
for s=1:subblocks
    hold on
    plot(py0{s}+(s-1)*size(X,1)/s,'.');title('Decentralized displacement estimate + Registered data');xlabel('time bins');ylabel('displacement');grid on;set(gca,'xlim',[1 length(py{s})]);
end
subplot(5,1,[2 5]);imagesc(log1p(flipud(minmax(Xr))));colormap(flipud(gray(256)));title('mean ptp vs. time');xlabel('time bins');ylabel('depth');

globalToc=toc(globalTic);

disp(['Total time of computation : ' num2str(globalToc/60) ' minutes.']);
disp(['Seconds of computation per second of recording: ' num2str(globalToc/length(T))]);