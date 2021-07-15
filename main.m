clear all
clc
close all

globalTic=tic;
%% parameters
time_resolution=1;
depth_resolution=1;
denoising_sigma = 0.1;
subsampling_level=log(1000)/1000;
robust_regression_sigma=1;
%% load data

%% NP2 - spike localizations + max amplitude + spike times (user provided)
% amps = readNPY('NP2_data/max_ptp.npy');
% depths = readNPY('NP2_data/optimized_z_position.npy');
% times = readNPY('NP2_data/spike_times.npy');times=times/30000;
% widths = readNPY('NP2_data/optimized_x_position.npy');

% amps = readNPY('NP2_data/max_ptp.npy');
% depths = readNPY('NP2_data/weighted_mean_z_position.npy');
% times = readNPY('NP2_data/spike_times.npy');times=times/30000;
% widths = readNPY('NP2_data/weighted_mean_x_position.npy');

% amps = readNPY('NP2_data/max_ptp.npy');
% depths = readNPY('NP2_data/all_cole_z_position.npy');
% times = readNPY('NP2_data/spike_times.npy');times=times/30000;
% widths = readNPY('NP2_data/all_cole_x_position.npy');

%% NP1 - spike localizations + max amplitude + spike times (user provided)

% amps = readNPY('NP1_data/max_ptp.npy');
% depths = readNPY('NP1_data/optimized_z_positions.npy');
% times = readNPY('NP1_data/spike_times.npy');times=times/30000;
% widths = readNPY('NP1_data/optimized_x_positions.npy');

% amps = readNPY('NP1_data/max_ptp.npy');
% depths = readNPY('NP1_data/mean_z_positions.npy');
% times = readNPY('NP1_data/spike_times.npy');times=times/30000;
% widths = readNPY('NP1_data/mean_x_positions.npy');

% amps = readNPY('NP1_data/max_ptp.npy');
% depths = readNPY('NP1_data/np1_all_denoised_z_cole.npy');
% times = readNPY('NP1_data/spike_times.npy');times=times/30000;
% widths = readNPY('NP1_data/np1_all_denoised_x_cole.npy');


%% simulated data
[depths,amps,times,widths,p0]=simulated_localizations(1001);



%% allocate bin sizes
T=floor(min(times)):time_resolution:ceil(max(times));
Ybins=floor(min(depths)):depth_resolution:ceil(max(depths));


%% generate image representations - depths
tic;
for t=1:length(T)-1
    data{t}(:,1)=depths(and(times>=T(t),times<=T(t+1)));
    data{t}(:,2)=amps(and(times>=T(t),times<=T(t+1)));
    data{t}(:,3)=widths(and(times>=T(t),times<=T(t+1)));
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

%% Poisson denoising
Xd=cheap_anscombe_denoising(X,'nlmeans',denoising_sigma);


%% decentralized motion estimation (ICASSP '21)
% generate "images"
for i=1:size(Xd,2)
    Ir{i}=Xd(:,i);
end
% generate pairwise displacement matrix
[~,Dy,Csub]=subsampled_pairwise_registration(Ir,subsampling_level,100,'');

% do robust regression to get the central estimate
py=psolver(Dy',robust_regression_sigma);py=py';

%% registration
data_reg=data;
for i=1:size(Xd,2)
    Xd_reg(:,i)=imtranslate(Xd(:,i),[0 py(i)]);
    data_reg{i}=data{i}+[py(i) 0 0];
end


globalToc=toc(globalTic);
disp(['Total time taken: ' num2str(globalToc) ' seconds. Time per 1 second of data: ' num2str(globalToc/size(Xd,2))]);

%% visualization
figure('units','normalized','outerposition',[0 0 1 0.4])
subplot(1,3,1)
imagesc([Xd]);title('Uncorrected raster');colormap(othercolor('Msunsetcolors'));xlabel('time(s)');ylabel('depth(um)');set(gca,'FontWeight','bold','FontSize',20,'TickLength',[0 0]);;colorbar
subplot(1,3,3)
imagesc([Xd_reg]);title('Registered raster');colormap(othercolor('Msunsetcolors'));xlabel('time(s)');ylabel('depth(um)');set(gca,'FontWeight','bold','FontSize',20,'TickLength',[0 0]);;colorbar
subplot(1,3,2)
plot(py,'m.');title('Displacement estimate');set(gca,'color','k');xlabel('time(s)');ylabel('y-displacement(um)');set(gca,'FontWeight','bold','FontSize',20,'TickLength',[0 0]);
