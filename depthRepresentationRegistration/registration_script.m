clear all
clc
close all


globalTic=tic;
%% parameters
time_resolution  = 1; % in seconds
depth_resolution = 1; % in microns
subblocks=10; % number of rigid blocks of motion
blocksize=1000; % in microns

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
lower_end=linspace(1,size(I{1},1)-blocksize,subblocks);
upper_end=linspace(blocksize,size(I{1},1),subblocks);
for s=1:subblocks
    blockcoor{s}=floor(lower_end(s)):ceil(upper_end(s));
    for t=1:length(I)
        Is{t,s}=I{t}(blockcoor{s});
    end
end

for s=1:subblocks
    [Dx{s},Dy{s},py{s},px{s},py0{s},px0{s}]=subsampled_pairwise_registration(Is(:,s),log(length(I))/length(I));
end


%% Undoing the translation
Xr=zeros(size(X));
counts=zeros(size(X));
tic
for t=1:length(I)
    for s=1:subblocks
        Xr(blockcoor{s},t)=Xr(blockcoor{s},t)+imtranslate(X(blockcoor{s},t),[0 -py{s}(t)]);
        counts(blockcoor{s},t)=counts(blockcoor{s},t)+1;
    end
    clc
    fprintf(['Translating data (' num2str(t) '/' num2str(length(T)-1) ')...\n']);
    fprintf(['\n' repmat('.',1,50) '\n\n'])
    for tt=1:round(t*50/(length(T)-1))
        fprintf('\b|\n');
    end
    TT=toc;
    disp(['Time elapsed (minutes): ' num2str(TT/60) ' Time remaining (minutes): ' num2str(((length(T)-1)-t)*(TT/t)*(1/60))]);
    
end

Xr=Xr./counts;
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
    plot(py{s},'LineWidth',1);title('Decentralized displacement estimate + Registered data');xlabel('time bins');ylabel('displacement');grid on;set(gca,'xlim',[1 length(py{s})]);
end
subplot(5,1,[2 5]);imagesc(log1p(flipud(minmax(Xr))));colormap(flipud(gray(256)));title('mean ptp vs. time');xlabel('time bins');ylabel('depth');

globalToc=toc(globalTic);

disp(['Total time of computation : ' num2str(globalToc/60) ' minutes.']);
disp(['Seconds of computation per second of recording: ' num2str(globalToc/length(T))]);