load('./Images/PETMRI5.mat');
load('res.mat');


% Generate Projection Data

N = length(img1); % Size of Image
NPROJ = 256; % # of Projection
ND = 256;

phi          = linspace(-pi/2,pi/2,NPROJ);
s            = linspace(-0.49,0.49,ND);
[S,Phi]      = meshgrid(s,phi);
%P            = Xraymat(S(:),Phi(:),N);

load('P')
fd='Bimodality_Demo/res/meta/';
fnames=dir([fd '*.bmp']);
Pt = P';
for index=1:15
    img1=ns(:,:,index);
    %img1=im2double(imread(fullfile(fd,fnames(index).name)));
    Of = P*img1(:);
    c = 1e-3*ones(size(Of));
    scale = 1e9;
    f = scale*imnoise((Of)/scale,'poisson');
    save(sprintf('%s%02d_f.mat',fd,index),'f');
    fimg = reshape(f,[NPROJ,ND])';
    cimg = reshape(c,[NPROJ,ND])';
    optsEM = struct('mit_out',100);
    [u1EM,ResEM] = EMRecon(P,Pt,f,c,N,img1,optsEM);
    u1EM=min(1,u1EM);
    disp([index,psnr(u1EM,img1)]);
    imwrite(fimg,sprintf('%s%02d_org.png',fd,index));
    imwrite(img1,sprintf('%s%02d_clean.png',fd,index));
    %figure;plot(ResEM);drawnow;
    % Generate Partial Fourier Data
end
