clear
global f c P Pt u1EM img1 optsInd;
addpath('./tool/');

optsInd = struct('mit_learn',100,'mit_inn',200,'mit_out',500,'alpha',0.001,'beta',0.00005,'gamma',0.00005,'rho',0.5);

rand('seed',3000);
randn('seed',3000);
fd='Bimodality_Demo/mri/meta/';
forg='Bimodality_Demo/mri/test/';
fres='Bimodality_Demo\mri';
load([fd,'/picks.mat'])
fnames=dir([forg '*.bmp']);
PatchSizeRowInd = 8;
PatchSizeColInd = 8;
StepSizeRowInd = 1;
StepSizeColInd = 1;
FilterTypeInd = 'dct';

N=256;
F2 = @(x) fft2(x);
F2T = @(x) real(ifft2(x));
% F2 = @(x) fft2(x)/N;
% F2T = @(x) N*ifft2(x,'symmetric');
st='';
%% Data Generation
for index=1:1:15
    
    
    img2=im2double(imread(fullfile(forg,fnames(index).name)));
    %Itemp=im2double(imread(fullfile(fd,fnames(index).name)));
    %B=F2(Itemp);
    load(sprintf('%s/%02d.mat',fd,index));
    Itemp=F2T(B);
    Itemp=max(0,min(1,Itemp));
    ps=psnr(Itemp,img2);
    err=norm(Itemp-img2,'fro')/norm(img2,'fro');
    imwrite(Itemp,sprintf('%s%02d_%.4f_%.4f_IFFT.png',forg,index,ps,err));
    st=sprintf('%s\n%.4f',st,ps);
end
fp=fopen([forg,'IFFTres.csv'],'w');
fprintf(fp,'%s',st)
fclose(fp);