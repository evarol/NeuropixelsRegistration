function [Dx,Dy]=subsampled_pairwise_registration(H,subsampling,resolution)
% decentralized registration
% Input:  H             - (T x 1) cell   -  input image/histogram representation of data
%         subsampling   - 1x1 scalar     - subsapling rate between 0 and 1
%         resolution    - 1x1 scalar     -  the subpixel resolution i.e. resolution = 100 --> 1/100th of pixel resolution
% Output: Dx,Dy         - (T x T) matrix -  pairwise displacement matrix along the x and y directions


if nargin<2
    % default parameters
    subsampling=1;
    resolution = 100;
end

if nargin<3
    resolution = 100;
end

t=0;

tic;
Dx=nan(length(H),length(H));
Dy=nan(length(H),length(H));
S=generate_random_tree(length(H),subsampling); S=or(S,S'); %% pre-allocate which subsampled registrations to perform -- if there are all zero rows, likely to kick an error
num_reg=round(sum(S(:))/2);
for i=1:length(H)
    for j=i:length(H)
        if S(i,j)==1
            [output, ~] = dftregistration(fft2(H{i}),fft2(H{j}),resolution); %% Code from Manuel Guizar-Sicairos
            xoffSet=output(4);
            yoffSet=output(3);
            Dx(i,j)=xoffSet;
            Dy(i,j)=yoffSet;
            Dx(j,i)=-Dx(i,j);
            Dy(j,i)=-Dy(i,j);
            
            t=t+1;
            clc
            fprintf(['Decentralized registration (' num2str(t) '/' num2str(num_reg) ')...\n']);
            fprintf(['\n' repmat('.',1,50) '\n\n'])
            for tt=1:round(t*50/num_reg)
                fprintf('\b|\n');
            end
            T=toc;
            disp(['Time elapsed (minutes): ' num2str(T/60) ' Time remaining (minutes): ' num2str((num_reg-t)*(T/t)*(1/60))]);
        end
    end
end



end