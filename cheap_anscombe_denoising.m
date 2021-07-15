function z_inverse_anscombe=cheap_anscombe_denoising(z,denoiser,sigma,comparison_window_size,search_window_size)
if nargin<3
    sigma=1;
    search_window_size=5;
    comparison_window_size=5;
end
if nargin<=2
    sigma=1;
    denoiser='nlmeans';
    search_window_size=5;
    comparison_window_size=5;
end
if nargin<4
    search_window_size=5;
    comparison_window_size=5;
end
minmax = @(x)((x-min(x(:)))./max(x(:)-min(x(:)))); %scales data to 0-1


% 1) gaussianizing poissonian data

z_anscombe=2*sqrt(minmax(z) + 3/8); %anscombe transformation making poissonian data gaussian

% 2) gaussian denoising
if strcmpi(denoiser,'bm3d');
    z_anscombe_denoised = RF3D(z_anscombe,sigma); %% BM3D denoising
elseif strcmpi(denoiser,'nlmeans');
    z_anscombe_denoised =imnlmfilt(z_anscombe,'DegreeOfSmoothing',sigma,'ComparisonWindowSize',comparison_window_size,'SearchWindowSize',search_window_size); %% NL means denoising
elseif strcmpi(denoiser,'bilateral');
    z_anscombe_denoised = imbilatfilt(z_anscombe,sigma,search_window_size);
end
% 3) inverse anscombe transform to make gaussian data poissonian again
z_inverse_anscombe = (1/4)*z_anscombe_denoised.^2 - 1/8 + (1/4)*sqrt(3/2)*(1./z_anscombe_denoised) - (11/8)*(1./z_anscombe_denoised).^2 + (5/8)*sqrt(3/2)*(1./z_anscombe_denoised).^3;

end