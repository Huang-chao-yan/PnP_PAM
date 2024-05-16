clear all
close all
global f c P Pt u1EM img1 optsInd;
addpath('./tool/');

rand('seed',3000);
randn('seed',3000);

%% Data Generation

load('./Images/PETMRI5.mat');
load('res.mat');
img1=ns(:,:,4);
% Generate Projection Data

N = length(img1); % Size of Image
NPROJ = 256; % # of Projection
ND = 256;

cd './EmissionTomography-PET&SPECT'
startup;
phi          = linspace(-pi/2,pi/2,NPROJ);
s            = linspace(-0.49,0.49,ND);
[S,Phi]      = meshgrid(s,phi);
%P            = Xraymat(S(:),Phi(:),N);

cd ..
load('P')

Pt = P';
Of = P*img1(:);
c = 1e-3*ones(size(Of));
scale = 2e10;
f = scale*imnoise((Of)/scale,'poisson');
fimg = reshape(f,[NPROJ,ND])';
cimg = reshape(c,[NPROJ,ND])';
optsEM = struct('mit_out',100);
disp('1')
[u1EM,ResEM] = EMRecon(P,Pt,f,c,N,img1,optsEM);
%figure;plot(ResEM);drawnow;
% Generate Partial Fourier Data

img2=im2double(imread('Bimodality_Demo\Bimodality_Demo\mri\test\02.bmp'));
% F2 = @(x) fft2(x)/N;
% F2T = @(x) N*ifft2(x,'symmetric');
F2 = @(x) fft2(x);
F2T = @(x) real(ifft2(x));

beams = 30;
y = MRImask(N,beams);

picks = ifftshift(y);
if picks(1,1) == 0
    picks(1,1) = 1;
end
ind = union(find(picks~=0),1);

FI = F2(img2);
sigma = 0.05*max(img2(:));
B = zeros(N,N);
B(ind) = FI(ind)+sigma*(randn(length(ind),1)+1i*randn(length(ind),1));
Bback=B;
Itemp = F2T(B);
imwrite(Itemp,'test.bmp');
%Itemp=im2double(imread('test.bmp'));
B=0;
load('Bimodality_Demo\mri\meta\02.mat')
% figure; imshow(img1,[]); colormap('jet')
% figure; imshow(fimg,[]); colormap('jet')
% figure; imshow(img2,[])
% figure; imshow(u1EM,[0 max(img1(:))]); colormap('jet'); title(['EM Method : Error = ', num2str(norm(u1EM-img1,'fro')/norm(img1,'fro'))]);
% figure; imshow(Itemp,[0 max(img2(:))]); title(['Zero Fill : Error = ', num2str(norm(Itemp-img2,'fro')/norm(img2,'fro'))]);

%% Data Driven Separate Reconstruction

PatchSizeRowInd = 8;
PatchSizeColInd = 8;
StepSizeRowInd = 1;
StepSizeColInd = 1;
FilterTypeInd = 'dct';

optsInd = struct('mit_learn',100,'mit_inn',200,'mit_out',500,'alpha',0.001,'beta',0.00005,'gamma',0.00005,'rho',0.5);

% PET
% gammaPET = 0.014;
% muPET = 2;
% [u1Ind,v1Ind,D1Ind,Res1Ind] = DtDvPETRec(f,c,P,Pt,N,gammaPET,muPET,PatchSizeRowInd,PatchSizeColInd,StepSizeRowInd,StepSizeColInd,FilterTypeInd,u1EM,img1,optsInd,0.001,0.00005,0.00005,0.5);
% Error1Ind = norm(u1Ind-img1,'fro')/norm(img1,'fro');
% figure; imshow(u1Ind,[0 max(img1(:))]); colormap('jet');  title(['Data Driven Tight Frame : Error = ', num2str(psnr(u1Ind,img1))]);

%IRCNN_PET
% gammaPET = 0.0025;
% muPET = 0.05;
% [u1Ind1,v1Ind1,D1Ind1,Res1Ind1] = IRCNN_DtDvPETRec(20,0.1,f,c,P,Pt,N,gammaPET,muPET,PatchSizeRowInd,PatchSizeColInd,StepSizeRowInd,StepSizeColInd,FilterTypeInd,u1EM,img1,optsInd,0.001,0.00005,0.00005,1);
% Error1Ind1 = norm(u1Ind1-img1,'fro')/norm(img1,'fro');
% figure; imshow(u1Ind1,[0 max(img1(:))]); colormap('jet');  title(['IRCNN Data Driven Tight Frame : Error = ', num2str(psnr(u1Ind1,img1))]);

%MWCNN
% [u1Ind2,v1Ind2,D1Ind2,Res1Ind2] = MWCNN_DtDvPETRec(24,0.6,f,c,P,Pt,N,0.002,2.8,PatchSizeRowInd,PatchSizeColInd,StepSizeRowInd,StepSizeColInd,FilterTypeInd,u1EM,img1,optsInd,0.001,0.00005,0.00005,0.5);
% Error1Ind2 = norm(u1Ind2-img1,'fro')/norm(img1,'fro');
% figure; imshow(u1Ind2,[0 max(img1(:))]); colormap('jet');  title(['MWCNN Data Driven Tight Frame : Error = ', num2str(psnr(u1Ind2,img1))]);


% Sparse MRI
gammaMRI = 0.01;
muMRI = 1;
[u2Ind,v2Ind,D2Ind,Res2Ind] = DtDvMRIRec(B,picks,gammaMRI,muMRI,PatchSizeRowInd,PatchSizeColInd,StepSizeRowInd,StepSizeColInd,FilterTypeInd,Itemp,img2,optsInd,0.001,0.00005,0.00005,0.5);
Error2Ind = norm(u2Ind-img2,'fro')/norm(img2,'fro');
figure; imshow(u2Ind,[0 max(img2(:))]); title(['Data Driven Tight Frame : Error = ', num2str(Error2Ind)]);

%% Wavelet Frame BiModality - Analysis (In replacement of JTV)

frame = 3;
Level = 1;
wLevel = 0.25;
kappa = 1;
lambdaAnal = 0.002;
optsAnal = struct('mit_out',10000,'mit_inn',1000,'u1_ini',u1EM,'u2_ini',Itemp);
[u1Anal,u2Anal,Res1Anal,Res2Anal] = WaveletFrameBiModality(f,c,P,Pt,N,B,picks,lambdaAnal,kappa,frame,Level,wLevel,img1,img2,optsAnal);

Error1Anal = norm(u1Anal-img1,'fro')/norm(img1,'fro');
Error2Anal = norm(u2Anal-img2,'fro')/norm(img2,'fro');

figure; imshow(u1Anal,[0 max(img1(:))]); colormap('jet'); title(['Joint Analysis : Error = ', num2str(Error1Anal)]);
figure; imshow(u2Anal,[0 max(img2(:))]); title(['Joint Analysis : Error = ', num2str(Error2Anal)]);

%% Wavelet Frame BiModality - Balanced

frame = 3;
Level = 1;
wLevel = 0.25;

kappa = 1;

mu1WF = 0.05;
mu2WF = 1;

lambdaWF = 0.00015;
optsWF = struct('mit_inn',1000,'mit_out',1000,'alpha1',0.001,'alpha2',0.001,'gamma',0.00005,'rho1',0.5,'rho2',0.5);
[u1WF,u2WF,v1WF,v2WF,Res1WF,Res2WF] = WaveletFrameBiModalityBalanced(f,c,P,Pt,N,B,picks,lambdaWF,kappa,mu1WF,mu2WF,frame,Level,wLevel,u1EM,Itemp,img1,img2,optsWF);

Error1WF = norm(u1WF-img1,'fro')/norm(img1,'fro');
Error2WF = norm(u2WF-img2,'fro')/norm(img2,'fro');

figure; imshow(u1WF,[0 max(img1(:))]); colormap('jet'); title(['Wavelet Frame BiModality : Error = ', num2str(Error1WF)]);
figure; imshow(u2WF,[0 max(img2(:))]); title(['Wavelet Frame BiModality : Error = ', num2str(Error2WF)]);

%% Data Driven BiModality

PatchSizeRow = 8;
PatchSizeCol = 8;
StepSizeRow = 1;
StepSizeCol = 1;
FilterType1 = 'dct';
FilterType2 = 'dct';

kappa = 1;

mu1 = 0.05;
mu2 = 1;

lambda = 0.003;
opts = struct('mit_learn',500,'mit_inn',500,'mit_out',1000,'alpha1',0.001,'alpha2',0.001,'beta1',0.00005,'beta2',0.00005,'gamma',0.00005,'rho1',0.5,'rho2',0.5);

[u1,u2,v1,v2,D1,D2,Res1,Res2] = DtDvBiModality(f,c,P,Pt,N,B,picks,lambda,kappa,mu1,mu2,PatchSizeRow,PatchSizeCol,StepSizeRow,StepSizeCol,FilterType1,FilterType2,u1EM,Itemp,img1,img2,opts);

Error1Fin = norm(u1-img1,'fro')/norm(img1,'fro');
Error2Fin = norm(u2-img2,'fro')/norm(img2,'fro');
figure; imshow(u1,[0 max(img1(:))]); colormap('jet'); title(['DDTF BiModality : Error = ', num2str(Error1Fin)]);
figure; imshow(u2,[0 max(img2(:))]); title(['DDTF BiModality : Error = ', num2str(Error2Fin)]);