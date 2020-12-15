function [X,geom,data,data_filter]=batch_extract(h5file,sample_rate,starttime,endtime,timeDS)
if nargin<5
    timeDS=1000;
end
start=[starttime*sample_rate+1 1];
count=[endtime*sample_rate-starttime*sample_rate 385];
data=h5read(h5file,'/recording',start,count)';
geom=h5read(h5file,'/geom');


% data=double(data(:,:))./std(double(data),[],2);
geom=geom';
geom=geom-min(geom,[],1);
geom=geom+1;


% y_map=exp(-pdist2((1:max(geom(:,2)))',geom(:,2)).^2/1000)>=0.9;
% x_map=exp(-pdist2((1:max(geom(:,1)))',geom(:,1)).^2/1000)>=0.9;
% y_map = y_map./sum(y_map,2);
% x_map = x_map./sum(x_map,2);
% My=y_map*movmax(double(data),121,2);
% Ny=y_map*movmin(double(data),121,2);
% X=imresize(My-Ny,[size(My,1) timeDS*(-starttime+endtime)],'nearest');
My=movmax(double(data),121,2);
Ny=movmin(double(data),121,2);
X=My-Ny;

if nargout==4
    data_filter=zeros(size(data));
    for t=1:size(data,1)
        data_filter(t,:)=bandpass(double(data(t,:)),[300 2000],sample_rate);
    end
end