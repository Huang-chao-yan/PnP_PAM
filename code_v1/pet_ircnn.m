clear
global f c P Pt u1EM img1 optsInd;
addpath('./tool/');
run D:\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn
% addpath(genpath('D:/MatlabFiles/MWCNN-master/MWCNN-master/.'))
addpath('func')

PatchSizeRow=8;
PatchSizeCol=8;
StepSizeRow=1;
StepSizeCol=1;
FilterType='dct';
fd='res/meta/';
load('P')
Pt = P';
%% Data Generation
index=1;

img1=im2double(imread(sprintf('%s%02d_clean.png',fd,index)));  % clean image
% Generate Projection Data

N = length(img1); % Size of Image
NPROJ = 256; % # of Projection
ND = 256;


phi          = linspace(-pi/2,pi/2,NPROJ);
s            = linspace(-0.49,0.49,ND);
[S,Phi]      = meshgrid(s,phi);

Of = P*img1(:);
c = 1e-3*ones(size(Of));
noiseLevel = 1e9;
f = noiseLevel*imnoise((Of)/noiseLevel,'poisson'); % degrade

fimg = reshape(f,[NPROJ,ND])';
cimg = reshape(c,[NPROJ,ND])';
optsEM = struct('mit_out',100);
[u1EM,ResEM] = EMRecon(P,Pt,f,c,N,img1,optsEM);
%% Parameters
optsInd = struct('mit_learn',300,'mit_inn',300,'mit_out',500,'alpha',0.001,'beta',0.00005,'gamma',0.00005,'rho',0.5);
scale=9;
xi=1;
%% Process image
tic
[u1Ind1,v1Ind1,D1Ind1,Res1Ind1] = OnlyIRCNN_DtDvPETRec(scale,xi,f,c,P,Pt,N,u1EM,img1,optsInd,0.001,0.5);
err=norm(u1Ind1-img1,'fro')/norm(img1,'fro')
ps=psnr(u1Ind1,img1)

toc
