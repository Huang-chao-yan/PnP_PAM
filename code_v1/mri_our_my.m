clear
global optsInd;
addpath('./tool/');
% run /home/extradisk/huangchaoyan/matconvnet-1.0-beta25/matlab/vl_setupnn
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
%load(sprintf('%s/%02d.mat',fd,index));
rand('seed',0);
%Itemp=F2T(B);
ratio = 0.5;
 MaskType = 1; % 1 for random mask; 2 for text mask
        switch MaskType
            case 1
                rand('seed',0);
                O = double(rand(size(img2)) > (1-ratio));
            case 2
                O = imread('TextMask256.png');
                O = double(O>128);
        end
        y= img2.* O;
       
        Itemp = F2T(y);
%% Parameters
scale=5;
xi=0.1;
lambda=0.00001;
mu=0.01;
%% Process image
tic;
[u2Ind,v2Ind,D2Ind,Res2Ind] = EST_MWCNN_DtDvMRIRec(scale,xi,y,O,lambda,mu,PatchSizeRowInd,PatchSizeColInd,StepSizeRowInd,StepSizeColInd,FilterTypeInd,Itemp,img2,optsInd,0.001,0.00005,0.00005,0.5);
err=norm(u2Ind-img2,'fro')/norm(img2,'fro')
ps=psnr(u2Ind,img2)
toc

