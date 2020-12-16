Dm=min(pdist2(coor,geom),[],2);
dist=unique(Dm(:));
for i=1:length(dist)
    template=[];
    target=[];
    for t=1:length(data)
        template=[template;I{t}(Dm==0)];
        target=[target;I{t}(Dm==dist(i))];
    end
    [~,beta(:,i)]=linhistmatch(target,template,200,'regular');
    i
end

vfield=zeros(size(Dm));
dfield=zeros(size(Dm));
for i=1:length(dist)
    vfield(Dm==dist(i))=beta(1,i);
    dfield(Dm==dist(i))=beta(2,i);
end

Sv=imgaussfilt(reshape(vfield,size(x')),2);
Dv=imgaussfilt(reshape(dfield,size(x')),2);
Dm=reshape(Dm,size(Sv));

for t=1:length(data)
    Ic{t}=I{t}.*Sv + Dv;t
end

minmax = @(x)((x-min(x(:)))./max(x(:)-min(x(:))));
figure
for t=1:length(data)
    subplot(1,2,1)
	hold on
    plot(Dm(:),I{t}(:),'.','MarkerSize',8);
    set(gca,'Ylim',[0 10]);xlabel('Distance to nearest probe (um)');ylabel('Intensity');title('Before vignetting correction');
    subplot(1,2,2)
    hold on
    plot(Dm(:),Ic{t}(:),'.','MarkerSize',8);
    set(gca,'Ylim',[0 10]);xlabel('Distance to nearest probe (um)');ylabel('Intensity');title('After vignetting correction');
end

figure
mind=min(size(Sv));
subplot(2,2,1);imagesc([Sv(1:mind,1:mind)]);colorbar;xlabel('y (um)');ylabel('x (um)');title('Multiplicative correction');axis square
subplot(2,2,2);imagesc([Dv(1:mind,1:mind)]);colorbar;xlabel('y (um)');ylabel('x (um)');title('Additive correction');axis square
subplot(2,2,3);imagesc([Dm(1:mind,1:mind)]);colorbar;xlabel('y (um)');ylabel('x (um)');title('Distance to nearest probe (um)');axis square
subplot(2,2,4);imagesc(ones(mind).*Sv(1:mind,1:mind)+Dv(1:mind,1:mind));colorbar;xlabel('y (um)');ylabel('x (um)');title('Voltage fall-off');axis square
% figure
% while 1==1
%     for t=1:length(data)
%         imagesc([I{t};Ic{t}]);drawnow
%     end
% end