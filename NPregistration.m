clear all
clc
close all

%%% NEUROPIXELS REGISTRATION --
%%% ----see vignettes for examples of different datasets and replace file paths

%% PARAMETERS
globalTic=tic;
timebins=200; %% set the number of time bins to use
mintime=0;%starting time point (in seconds)
maxtime=1999;%ending time point (in seconds)
timestep=1; %seconds (how big should each time bin be)
sample_rate=30000; %%Set the sampling rate of recording
threshold=6;
L=32;
sigma=0;
%% DATA FORMAT
dataset='NP-binary2'; %types of data format (NP-binary,NP-H5,EPHYS)



%% helper functions
times=round(linspace(mintime,maxtime,timebins));
ptp=@(x)(movmax(x,25,2)-movmin(x,25,2));
vec=@(x)(x(:));
bp = @(x)(reshape(bandpass(vec(double(x)'),[300 2000],30000),size(x'))');
thresh = @(x,t)(x.*(x>t));
minmax = @(x)((x-min(x(:)))./max(x(:)-min(x(:))));
[b,a] = butter(3,[300 2000]/(30000/2));
bp2 = @(x)(reshape(filter(b,a,vec(double(x)')),size(x'))');

%% DATA LOADING
%% see different examples of data formats supported)
if strcmpi(dataset,'NP-H5');
    
    %% time bin loading - h5 files
    bestsnr=0;
    tic
    for t=1:length(times)
        
        [X,geom,data{t}]=batch_extract('\Users\Erdem\Dropbox\Projects\spike_drift\neuropixel_data.h5',sample_rate,times(t),times(t)+timestep,1000);
        
        
        
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
    for t=1:length(times)
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

if strcmpi(dataset,'NP-binary2')
    %% time bin loading - binary files
    fileID = fopen('/Users/erdem/Downloads/cortexlab-drift-dataset1.bin','r');
    dataset_info=load('/Users/erdem/Downloads/NP2_kilosortChanMap.mat');
    geom(:,1)=dataset_info.xcoords;
    geom(:,2)=dataset_info.ycoords;
    bestsnr=0;
    tic
    for t=1:length(times)
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
    dataset_info=load('/Users/erdem/Downloads/buz32chMap.mat');
    for t=1:8
        [tmp, timestamps, info] = load_open_ephys_data(['/Users/erdem/Downloads/2019-11-18_16-27-36_HR46_R0_001/100_CH' num2str(t) '.continuous']);
        if t==1
            data0=zeros(8,size(tmp,1));
        end
        data0(t,:)=tmp';
        i
        
        
        clc
        fprintf(['Loading time bins (' num2str(t) '/' num2str(32) ')...\n']);
        fprintf(['\n' repmat('.',1,50) '\n\n'])
        for tt=1:round(t*50/32)
            fprintf('\b|\n');
        end
        T=toc;
        disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((32-t)*(T/t)*(1/60))]);
        
    end
    
    rec_length=floor(size(data0,2)/24000);
    tic
    for t=1:rec_length
        data{t}=data0(:,(t-1)*24000+1:t*24000);
        clc
        fprintf(['Loading time bins (' num2str(t) '/' num2str(rec_length) ')...\n']);
        fprintf(['\n' repmat('.',1,50) '\n\n'])
        for tt=1:round(t*50/rec_length)
            fprintf('\b|\n');
        end
        T=toc;
        disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((rec_length-t)*(T/t)*(1/60))]);
    end
    geom(:,1)=dataset_info.xcoords(1:8,:);
    geom(:,2)=dataset_info.ycoords(1:8,:);
    
    clear data0;
end


%% MAIN ROUTINE
%% Filtering / background removal



%% feature extraction
nhood=L;
geom=geom-min(geom,[],1)+1;
[x,y]=meshgrid((min(geom(:,1)):max(geom(:,1))),(min(geom(:,2)):max(geom(:,2))));
coor=[vec(permute(x,[2 1])) vec(permute(y,[2 1]))];
M=mapping_matrix(geom,coor,'krigging',1,sigma,L);
tic;
for t=1:length(data)
    
    
    A=thresh(ptp(decorrelate(bp2(data{t}),2)),threshold);
    E{t}=zeros(size(A,1),1);
    E{t}(any(A>0,2))=sum(A(any(A>0,2),:),2)./sum(A(any(A>0,2),:)>0,2);
    E{t}=max(E{t}-threshold,0);
    
    %     A=thresh(ptp(bp2(decorrelate(single(data{t}),2))),threshold);
    %     A(A==0)=nan;
    %     E=nanmean(A,2);E(isnan(E))=0;
    %     E=max(E-threshold,0);
    
    I{t}=max(reshape(M*E{t},size(x')),0);
    clc
    fprintf(['Generating images (' num2str(t) '/' num2str(length(times)) ')...\n']);
    fprintf(['\n' repmat('.',1,50) '\n\n'])
    for tt=1:round(t*50/length(times))
        fprintf('\b|\n');
    end
    T=toc;
    disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((length(times)-t)*(T/t)*(1/60))]);
end


%% decentralized displacement estimate

[Dx,Dy,py,px,cmax]=pairwise_reg(I,1,100);


%% Vignetting correction using histogram normalization

%% Unregistered spikes and histograms for the purpose of visualization
for t=1:length(data)
    H{t}=quantile(decorrelate(bp2(data{t}),2),linspace(0,1,100),2); %% histograms
    timebindata=decorrelate(bp2(data{t}),2);
    P=ptp(timebindata);
    for i=1:size(data{t},1)
        spikes{i,t,1}=find(P(i,:)>=6); % spike times
        spikes{i,t,2}=P(i,spikes{i,t,1}); % spike amplitude
    end
end

%% Registered spikes and histograms for the purpose of visualization
for t=1:length(data)
    Mi=interpolation_matrix(geom,[px(t) py(t)],'krigging',1,0,L);
    timebindata=decorrelate(Mi*bp2(data{t}),2);
    P=ptp(timebindata);
    Hr{t}=quantile(timebindata,linspace(0,1,100),2);
    for i=1:size(data{t},1)
        spikes2{i,t,1}=find(P(i,:)>=6); % spike times
        spikes2{i,t,2}=P(i,spikes2{i,t,1}); % spike amplitude
    end
    t
end


%% Histogram normalization
template=0;
for t=1:length(data)
    template=template+Hr{t}/length(data);
end
for t=1:length(data)
    Mi=interpolation_matrix(geom,[px(t) py(t)],'krigging',1,0,L);
    timebindata=decorrelate(Mi*bp2(data{t}),2);
    
    for i=1:size(Hr{t},1)
        [Hrc{t}(i,:),beta]=linhistmatch(Hr{t}(i,:),template(i,:),10,'regular'); % corrected histograms
        data_reg{t}(i,:)=timebindata(i,:)*beta(1) + beta(2); %%%%%%%%%<<<----REGISTERED + HISTOGRAM CORRECTED DATA
        P=ptp(timebindata(i,:)*beta(1)+beta(2));
        spikes3{i,t,1}=find(P>=6); % spike times
        spikes3{i,t,2}=P(spikes3{i,t,1}); % spike amplitude
        [t i]
    end
end

%% Linear regressing out effect of displacement

clear beta
for i=1:size(Hr{t},1)
    for t=1:length(data)
        G(:,t)=Hr{t}(i,:)';
    end
    beta{i}=pinv([vec(ones(size(G,1),1)*py) vec(ones(size(G,1),1)*px) ones(size(vec(ones(size(G,1),1)*py)))])*vec(G);
    Gregressed = reshape(vec(G) - [vec(ones(size(G,1),1)*py) vec(ones(size(G,1),1)*px)]*beta{i}(1:2),size(G));
    for t=1:length(data)
        Hr2{t}(i,:)=Gregressed(:,t);
    end
    
end


for t=1:length(data)
Mi=interpolation_matrix(geom,[px(t) py(t)],'krigging',1,0,32);
timebindata=decorrelate(Mi*bp2(data{t}),2);
for i=1:size(data{t},1)
P=ptp(timebindata(i,:) - beta{i}(1)*py(t) - beta{i}(2)*px(t));
spikes4{i,t,1}=find(P>=6);
spikes4{i,t,2}=P(spikes4{i,t,1});
end
t
end


%% visualizations


%% stacking maximum PTP per time bin wide to side for each channel
clear X*
for t=1:length(data)
    X(:,t)=max(H{t},[],2);
    Xr(:,t)=max(Hr{t},[],2);
    Xr2(:,t)=max(Hr2{t},[],2);
    Xr3(:,t)=max(Hrc{t},[],2);
end




%% visualization of maximum PTP
Z=[X Xr Xr2 Xr3];   
figure('units','normalized','outerposition',[0 0 1 1])
imagesc(Z);colorbar
hold on
plot([0.5+size(X,2) 0.5+size(X,2)],[0.5 0.5+size(X,1)],'w-','LineWidth',2);
plot([0.5+2*size(X,2) 0.5+2*size(X,2)],[0.5 0.5+size(X,1)],'w-','LineWidth',2);
plot([0.5+3*size(X,2) 0.5+3*size(X,2)],[0.5 0.5+size(X,1)],'w-','LineWidth',2);
text(10,10,'1) Unregistered/uncorrected','Color','w','Fontweight','bold');
text(10+size(X,2),10,'2) Registered/uncorrected','Color','w','Fontweight','bold');
text(10+2*size(X,2),10,'3) Registered/regress out d correction','Color','w','Fontweight','bold');
text(10+3*size(X,2),10,'4) Registered/channel histogram correction','Color','w','Fontweight','bold');
ylabel('Channels');xlabel('Time (s)');
set(gca,'FontWeight','bold');
export_fig('histogram_corrected_registration.png');




%% selecting a specific channel to look at
title('Right click on a depth to visualize amplitudes closer...');
disp('Right click on a depth to visualize amplitudes closer...');
[xcoor,ycoor]=getpts;
offset=3;

ycoor=round(ycoor(end));

[~,midx]=max(mean(Z(ycoor-offset:ycoor+offset,:),2),[],1);

ycoor=ycoor+midx-offset;


%% channel specific zoom
figure('units','normalized','outerposition',[0 0 1 2/3])
subplot(3,1,1)
imagesc(Z(ycoor-offset:ycoor+offset,:));
hold on
plot([0.5 size(Z,2)+0.5],[offset offset],'r--','LineWidth',1);

plot([0.5+size(X,2) 0.5+size(X,2)],[0.5 0.5+offset*2],'w-','LineWidth',2);
plot([0.5+2*size(X,2) 0.5+2*size(X,2)],[0.5 0.5+offset*2],'w-','LineWidth',2);
plot([0.5+3*size(X,2) 0.5+3*size(X,2)],[0.5 0.5+offset*2],'w-','LineWidth',2);
text(10,1,'1) Unregistered/uncorrected','Color','w','Fontweight','bold');
text(10+size(X,2),1,'2) Registered/uncorrected','Color','w','Fontweight','bold');
text(10+2*size(X,2),1,'3) Registered/regress out d correction','Color','w','Fontweight','bold');
text(10+3*size(X,2),1,'4) Registered/channel histogram correction','Color','w','Fontweight','bold');
ylabel('Neighboring Channels');xlabel('Time (s)');
set(gca,'FontWeight','bold');
title(['Channel ' num2str(ycoor) ],'FontSize',20,'FontWeight','bold');
subplot(3,1,2);
plot(Z(ycoor,:),'.');grid on
hold on
plot([0.5+size(X,2) 0.5+size(X,2)],[0 max(Z(ycoor,:))+0.5],'k-','LineWidth',2);
plot([0.5+2*size(X,2) 0.5+2*size(X,2)],[0 max(Z(ycoor,:))+0.5],'k-','LineWidth',2);
plot([0.5+3*size(X,2) 0.5+3*size(X,2)],[0 max(Z(ycoor,:))+0.5],'k-','LineWidth',2);
text(10,max(Z(ycoor,:)),['1) Unregistered/uncorrected - corr(x,d)=' num2str(corr(X(ycoor,:)',py'))],'Color','k','Fontweight','bold');
text(10+size(X,2),max(Z(ycoor,:)),['2) Registered/uncorrected - corr(x,d)=' num2str(corr(Xr(ycoor,:)',py'))],'Color','k','Fontweight','bold');
text(10+2*size(X,2),max(Z(ycoor,:)),['3) Registered/regress out d correction - corr(x,d)=' num2str(corr(Xr2(ycoor,:)',py'))],'Color','k','Fontweight','bold');
text(10+3*size(X,2),max(Z(ycoor,:)),['4) Registered/channel histogram correction - corr(x,d)=' num2str(corr(Xr3(ycoor,:)',py'))],'Color','k','Fontweight','bold');
ylabel('max PTP');xlabel('Time (s)');
set(gca,'FontWeight','bold');
subplot(3,1,3)
hold on
maxB=0;
for t=1:length(data)
A=spikes{ycoor,t,1}+30000*(t-1);
B=spikes{ycoor,t,2}-6;if ~isempty(B);maxB=nanmax(maxB,nanmax(B));end
plot(A(B>0),B(B>0),'b.','MarkerSize',0.5);
end
for t=1:length(data)
A=spikes2{ycoor,t,1}+30000*(t-1);
B=spikes2{ycoor,t,2}-6;if ~isempty(B);maxB=nanmax(maxB,nanmax(B));end
plot(A(B>0)+30000*200,B(B>0),'b.','MarkerSize',0.5);
end
for t=1:length(data)
A=spikes4{ycoor,t,1}+30000*(t-1);
B=spikes4{ycoor,t,2}-6;if ~isempty(B);maxB=nanmax(maxB,nanmax(B));end
plot(A(B>0)+2*30000*200,B(B>0),'b.','MarkerSize',0.5);
end
for t=1:length(data)
A=spikes3{ycoor,t,1}+30000*(t-1);
B=spikes3{ycoor,t,2}-6;if ~isempty(B);maxB=nanmax(maxB,nanmax(B));end
plot(A(B>0)+3*30000*200,B(B>0),'b.','MarkerSize',0.5);
end
grid on
plot([0.5+30000*200 0.5+30000*200],[0 maxB+0.5],'k-','LineWidth',2);
plot([0.5+2*30000*200 0.5+2*30000*200],[0 maxB+0.5],'k-','LineWidth',2);
plot([0.5+3*30000*200 0.5+3*30000*200],[0 maxB+0.5],'k-','LineWidth',2);
text(30000*10,maxB,['1) Unregistered/uncorrected'],'Color','k','Fontweight','bold');
text(30000*10+200*30000,maxB,['2) Registered/uncorrected'],'Color','k','Fontweight','bold');
text(30000*10+2*200*30000,maxB,['3) Registered/regress out d correction'],'Color','k','Fontweight','bold');
text(30000*10+3*200*30000,maxB,['4) Registered/channel histogram correction'],'Color','k','Fontweight','bold');
ylabel('PTP');xlabel('Time (s)');
set(gca,'xlim',[0 30000*200*4],'ylim',[0 maxB]);
export_fig(['channel_' num2str(ycoor) '.png']);






%% displacement estimate fig

figure
subplot(1,2,1);
plot(py-py(1),'LineWidth',2,'Color',[0.5 0 0.5 0.5]);xlabel('Time bins');ylabel('Displacement');title('Y-Displacement estimate');grid on;

subplot(1,2,2);
plot(px-px(1),'LineWidth',2,'Color',[0 0.5 0.5 0.5]);xlabel('Time bins');ylabel('Displacement');title('X-Displacement estimate');grid on;

totalTime=toc(globalTic);

disp(['Total time: ' num2str(totalTime/60) ' minutes. ' num2str(totalTime/length(data)) ' seconds per one-second time bin of data.']);

