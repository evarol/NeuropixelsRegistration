clear all
clc
close all

%%% NEUROPIXELS REGISTRATION --
%%% ----see vignettes for examples of different datasets and replace file paths

%% PARAMETERS
globalTic=tic;
timebins=1000; %% set the number of time bins to use
mintime=0;%starting time point (in seconds)
maxtime=999;%ending time point (in seconds)
timestep=1; %seconds (how big should each time bin be)
sample_rate=30000; %%Set the sampling rate of recording


%% DATA FORMAT
dataset='NP-binary'; %types of data format (NP-binary,NP-H5,EPHYS)



%% helper functions
ptp=@(x)(movmax(x,121,2)-movmin(x,121,2));
vec=@(x)(x(:));
times=round(linspace(0,999,timebins));


%% DATA LOADING
%% see different examples of data formats supported)
if strcmpi(dataset,'NP-H5');
    
    %% time bin loading - h5 files
    bestsnr=0;
    tic
    for t=1:length(times)-1
        [X,geom,data{t}]=batch_extract('erdem_data.h5',sample_rate,times(t),times(t)+timestep,1000);
        
        [maxsnr,maxchan]=max(quantile(X,0.9,2)-quantile(X,0.1,2));
        
        if maxsnr>bestsnr
            max_chan_bin=[t maxchan];
        end
        clc
        fprintf(['Loading time bins (' num2str(t) '/' num2str(length(times)) ')...\n']);
        fprintf(['\n' repmat('.',1,50) '\n\n'])
        for tt=1:round(t*50/length(times))
            fprintf('\b|\n');
        end
        T=toc;
        disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((length(times)-t)*(T/t)*(1/60))]);
        
    end
    
end

if strcmpi(dataset,'NP-binary')
    %% time bin loading - binary files
    fileID = fopen('/Users/erdem/Downloads/pacman-task_c_191202_neu_001_CAR.bin','r');
    dataset_info=load('/Users/erdem/Downloads/neuropixels_primateDemo128_chanMap.mat');
    geom(:,1)=dataset_info.xcoords;
    geom(:,2)=dataset_info.ycoords;
    bestsnr=0;
    tic
    for t=1:length(times)-1
        fseek(fileID,128*30000*times(t),'bof');
        data{t} = fread(fileID, [128 30000], '*int16');
        clc
        fprintf(['Loading time bins (' num2str(t) '/' num2str(length(times)) ')...\n']);
        fprintf(['\n' repmat('.',1,50) '\n\n'])
        for tt=1:round(t*50/length(times))
            fprintf('\b|\n');
        end
        T=toc;
        disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((length(times)-t)*(T/t)*(1/60))]);
        
    end
    fclose(fileID);
end

if strcmpi(dataset,'NP-binary2')
    %% time bin loading - binary files
    fileID = fopen('/Users/erdem/Downloads/cortexlab-drift-dataset1.bin','r');
    dataset_info=load('/Users/erdem/Downloads/NP2_kilosortChanMap.mat');
    geom(:,1)=dataset_info.xcoords;
    geom(:,2)=dataset_info.ycoords;
    bestsnr=0;
    tic
    for t=1:length(times)-1
        fseek(fileID,length(dataset_info.chanMap)*30000*times(t),'bof');
        data{t} = fread(fileID, [length(dataset_info.chanMap) 30000], '*int16');
        clc
        fprintf(['Loading time bins (' num2str(t) '/' num2str(length(times)) ')...\n']);
        fprintf(['\n' repmat('.',1,50) '\n\n'])
        for tt=1:round(t*50/length(times))
            fprintf('\b|\n');
        end
        T=toc;
        disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((length(times)-t)*(T/t)*(1/60))]);
        
    end
    fclose(fileID);
end


if strcmpi(dataset,'EPHYS');
    
    %% time bin loading - EPHYS FILES
    bestsnr=0;
    tic
    for t=1:32
        for i=1:32
            [tmp, timestamps, info] = load_open_ephys_data(['/Users/erdem/Downloads/2019-11-18_16-27-36_HR46_R0_001/100_CH' num2str(i) '.continuous']);
            if i==1
                data=zeros(32,size(tmp,1));
            end
            data(i,:)=tmp';
            i
        end
        
        
        clc
        fprintf(['Loading time bins (' num2str(t) '/' num2str(32) ')...\n']);
        fprintf(['\n' repmat('.',1,50) '\n\n'])
        for tt=1:round(t*50/32)
            fprintf('\b|\n');
        end
        T=toc;
        disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((32-t)*(T/t)*(1/60))]);
        
    end
    
end


%% MAIN ROUTINE
%% background removal

tic
for t=1:length(data)
    
    data_denoised{t}=sinkhorn_denoise(ptp(data{t}),0.95,2);
    clc
    fprintf(['Background removing (' num2str(t) '/' num2str(length(times)) ')...\n']);
    fprintf(['\n' repmat('.',1,50) '\n\n'])
    for tt=1:round(t*50/length(times))
        fprintf('\b|\n');
    end
    T=toc;
    disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((length(times)-t)*(T/t)*(1/60))]);
end


%% featurization
M=mapping_matrix(geom,[repmat(mean(geom(:,1)),size((min(geom(:,2)):max(geom(:,2)))',1),1) (min(geom(:,2)):max(geom(:,2)))'],'krigging',1,0.1,28);
H=zeros(size(M,1),length(data));
tic
for t=1:length(data)
    
    %     H(:,t)=M*max(log(max(double(data_denoised{t}),[],2)),0);
    H(:,t)=sum(max(M*double(data_denoised{t}),0),2);
    clc
    fprintf(['Feature generating (' num2str(t) '/' num2str(length(times)) ')...\n']);
    fprintf(['\n' repmat('.',1,50) '\n\n'])
    for tt=1:round(t*50/length(times))
        fprintf('\b|\n');
    end
    T=toc;
    disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((length(times)-t)*(T/t)*(1/60))]);
end


%% decentralized displacement estimate

H0=H;
minmax = @(x)((x-min(x(:)))./max(x(:)-min(x(:))));
H=double((minmax(H)>=0.01));

X=spatial_icdf(cumsum(H,1)./sum(H,1),linspace(0,1,40));

D=nan(size(X,2));
C=nan(size(X,2));
for i=1:size(X,2)
    D(i,:)=mean(X(:,i)-X,1)';
    C(i,:)=var(X(:,i)-X,[],1)';
end

p=mean(D-nanmean(D,2));
Dcorr=D;
% Dcorr(minmax(C)>=graythresh(minmax(C)))=nan;
Dcorr(minmax(C)>=graythresh(minmax(C)))=nan;
pcorr=nanmean(Dcorr-nanmean(Dcorr,2));
p=-(p-mean(p));
pcorr=-(pcorr-mean(pcorr));
close all
subplot(2,2,1)
imagesc(flipud(H.*H0));xlabel('Time bins (1s)');ylabel('Y-position (um)');title('Features');colormap(othercolor('Blues9'));
subplot(2,2,2)
imagesc(D);xlabel('Time bins (1s)');ylabel('TIme bins (1s)');title('Pairwise Displacement estimates');colormap(othercolor('BuDRd_12'));colorbar;axis square
subplot(2,2,4)
imagesc(Dcorr);xlabel('Time bins (1s)');ylabel('TIme bins (1s)');title('Pairwise Displacement estimates (error corrected)');colormap(othercolor('BuDRd_12'));colorbar;axis square
subplot(2,2,3)
plot(p);hold on;plot(imgaussfilt(p,20),'LineWidth',2);plot(pcorr);plot(imgaussfilt(pcorr,20),'LineWidth',2);legend({'Displacement estimate','Smoothed displacement estimate','Error corrected displacement estimate','Smoothed error corrected estimate'});xlabel('Time bin (1s)');ylabel('Displacement (um)');title('Global displacement estimate');


totalTime=toc(globalTic);

disp(['Total time: ' num2str(totalTime/60) ' minutes. ' num2str(totalTime/length(data)) ' seconds per one-second time bin of data.']);
