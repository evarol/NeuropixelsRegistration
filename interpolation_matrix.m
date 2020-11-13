function M=interpolation_matrix(geom,u,type,sigma,sigma_l,l)
if nargin<5
    sigma_l=1;
    l=0.1;
end
M=zeros(size(geom,1),size(geom,1));
if strcmpi(type,'bilinear');
    
    geom_disp=geom+u;
   
    for i=1:size(geom_disp,1)
        D=pdist2(geom_disp(i,:),geom);
        % 4 corners
        [~,idx]=sort(D,'ascend','MissingPlacement','last');
        idx=idx(1:4);
        dist=pdist2(geom_disp(i,:),geom(idx,:));
        if any(dist==0)
            M(i,idx(find(dist==0)))=1;
        else
            for j=1:4
                M(i,idx(j))=(1/dist(j))/sum(1./dist);
            end
        end
            
    end
    
    
    
end

if strcmpi(type,'gaussian')
   
    geom_disp=geom+u;
    M=exp(-pdist2(geom_disp,geom).^2/(2*sigma^2));
    M=M./sum(M,2);
    M(isnan(M))=0;
end

if strcmpi(type,'krigging')
    geom_disp=geom+u;
    K1=sigma*exp(-(squareform(pdist(geom)).^2)/(2*l^2)) + sigma_l*eye(size(geom,1));
    K2=sigma*exp(-((pdist2(geom_disp,geom)).^2)/(2*l^2));
    M=K2/K1;

end