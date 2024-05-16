load('./Images/PETMRI5.mat');
load('res.mat');


% Generate Projection Data

N = 256; % Size of Image



%P            = Xraymat(S(:),Phi(:),N);


fd='Bimodality_Demo/mri/test/';
fout='Bimodality_Demo/mri/meta/';
fnames=dir([fd '*.bmp']);

N=256;
F2 = @(x) fft2(x);
F2T = @(x) real(ifft2(x));
% F2 = @(x) fft2(x)/N;
% F2T = @(x) N*ifft2(x,'symmetric');
beams = 15;
y = MRImask(N,beams);
picks = ifftshift(y);
if picks(1,1) == 0
    picks(1,1) = 1;
end
ind = union(find(picks~=0),1);

save(sprintf('%s/picks.mat',fout),'picks');
for index=1:15
    
    img2=im2double(imread(fullfile(fd,fnames(index).name)));
    [h,w,c]=size(img2);
    if c>1
        img2=rgb2gray(img2);
        imwrite(img2,fullfile(fd,fnames(index).name))
    end
    
    FI = F2(img2);
    sigma = 0.05*max(img2(:));
    B = zeros(N,N);
    B(ind) = FI(ind)+sigma*(randn(length(ind),1)+1i*randn(length(ind),1));
    Itemp = F2T(B);
    %imshow(Itemp)
    name=split(fnames(index).name,'.');
    imwrite(Itemp,sprintf('%s/%s.bmp',fout,char(name(1))))
    save(sprintf('%s/%s.mat',fout,char(name(1))),'B')
    %figure;plot(ResEM);drawnow;
    % Generate Partial Fourier Data
end
