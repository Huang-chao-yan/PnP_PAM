clear
global optsInd;
addpath('./tool/');
run D:\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn
optsInd = struct('mit_learn',100,'mit_inn',200,'mit_out',500,'alpha',0.001,'beta',0.00005,'gamma',0.00005,'rho',0.5);

fd='mri/meta/';
forg='mri/test/';
fres='mri';
load([fd,'/picks.mat'])
fnames=dir([forg '*.bmp']);
PatchSizeRowInd = 8;
PatchSizeColInd = 8;
StepSizeRowInd = 1;
StepSizeColInd = 1;
FilterTypeInd = 'dct';

F2 = @(x) fft2(x);
F2T = @(x) real(ifft2(x));

%% Data Generation
index=1;

img2=im2double(imread(fullfile(forg,fnames(index).name)));
load(sprintf('%s/%02d.mat',fd,index));

Itemp=F2T(B);
%% Parameters
scale=5;
xi=0.1;
%% Process image

tic;
%err=testEST_MWCNN([xi,lambda,mu,0.001,0.00005,0.00005,0.5,models],index);
[u2Ind,v2Ind,D2Ind,Res2Ind] = OnlyIRCNN_DtDvMRIRec(scale,xi,B,picks,Itemp,img2,optsInd,0.001,0.5);

err=norm(u2Ind-img2,'fro')/norm(img2,'fro')
ps=psnr(u2Ind,img2)

toc