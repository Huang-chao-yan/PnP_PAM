%0.1203   12.0000    0.1900    0.0006    0.100 | 0.001,0.005,0.005,1
%0.1283   24.0000    0.6000    0.0020    2.8000  26.6724
%0.1290   73.0000    0.4000    0.0020    3.5000 26.6301
%0.1288   65.0000    0.5000    0.0020    2.0000 26.6428
%0.1645   60.0000    0.3000    0.0030    3.5000 26.3633 1/20
%0.1649   60.0000    0.3000    0.0030    3.0000 27.3091 4/20
% es=[];
% %betas=0.00005:0.01:0.1;
% rhos=0.1:0.1:5;
% for rho=rhos
%     %optsInd = struct('mit_learn',500,'mit_inn',500,'mit_out',1000,'alpha',0.1,'beta',0.0005,'gamma',0.005,'rho',rho);
%     [u1Ind2,v1Ind2,D1Ind2,Res1Ind2] =MWCNN_DtDvPETRec(12,0.21,f,c,P,Pt,N,0.0005,0.05,PatchSizeRowInd,PatchSizeColInd,StepSizeRowInd,StepSizeColInd,FilterTypeInd,u1EM,img1,optsInd,0.001,0.005,0.005,rho);
%     Error1Ind = norm(u1Ind2-img1,'fro')/norm(img1,'fro');
%     es(end+1)=psnr(u1Ind2,img1);
%     plot(es),drawnow;
% end
%%
clear
global f c P Pt u1EM img1 optsInd;
addpath('./tool/');

rand('seed',3000);
randn('seed',3000);
fd='Bimodality_Demo/res/meta/';
load('P')
Pt = P';
st='';
%% Data Generation
for index=1:1:15
    
    
    img1=im2double(imread(sprintf('%s%02d_clean.png',fd,index)));
    % Generate Projection Data
    
    N = length(img1); % Size of Image
    NPROJ = 256; % # of Projection
    ND = 256;
    
    
    phi          = linspace(-pi/2,pi/2,NPROJ);
    s            = linspace(-0.49,0.49,ND);
    [S,Phi]      = meshgrid(s,phi);
    %P            = Xraymat(S(:),Phi(:),N);
    
    
    
    Of = P*img1(:);
    c = 1e-3*ones(size(Of));
    load(sprintf('%s%02d_f.mat',fd,index));
    fimg = reshape(f,[NPROJ,ND])';
    cimg = reshape(c,[NPROJ,ND])';
    optsEM = struct('mit_out',100);
    [u1EM,ResEM] = EMRecon(P,Pt,f,c,N,img1,optsEM);
    %figure;plot(ResEM);drawnow;
    % Generate Partial Fourier Data
    u1EM=max(0,min(1,u1EM));
    ps=psnr(u1EM,img1);
    err=norm(u1EM-img1,'fro')/norm(img1,'fro');
    st=sprintf('%s\n%.4f',st,ps);
    disp(st)
    imwrite(u1EM,sprintf('%sslice%02d_%.4f_%.4f_EMRecon.png',fd,index,ps,err))
end
fp=fopen([fd,'EMres.csv'],'w');
fprintf(fp,'%s',st)
fclose(fp);