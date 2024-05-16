% Setup.m
%
  clear all, close all, randn('state',0);   %  Reset random number generator to initial state.
% -------------------------------------------------------------
% Generate true image and forward model
% To generate matrix, the following code was used:
% -------------------------------------------------------------
  nphi=128; ns=128; n = 128;
  %%% Load true target.
%   [object1,object2]=MakeTarget(n);
%   Emission = 500*(object1+object2);
%   SNR_flag = input('Enter desired SNR (5, 10, or 20). ');
%   if SNR_flag == 5
%       Emission = .5*Emission;
%   elseif SNR_flag == 10
%       Emission = 2*Emission;
%   elseif SNR_flag == 30
%       Emission = 8*Emission;
%   end
  Emission = 500*phantom(n);
  xtrue = Emission;    
    
  model_flag   = input('Enter 1 for PET and 2 for SPECT. ');
  phi          = linspace(-pi/2,pi/2,nphi);
  s            = linspace(-0.49,0.49,ns);
  [S,Phi]      = meshgrid(s,phi);
  A0           = Xraymat(S(:),Phi(:),n);
  if model_flag == 1
      Absorption = zeros(n,n);
      W = exp(-A0*Absorption(:)); 
      A = spdiags(W,0,ns*nphi,ns*nphi)*A0; 
  elseif model_flag == 2
      Absorption   = (Emission > 0);
      A = SPECTmat(S(:),Phi(:),n,Absorption,A0);
  end 
  Aparams.A    = A;
  Aparams.nphi = nphi;
  Aparams.ns   = ns;
  Aparams.N    = n;
  
%-------------------------------------------------------------------
% Generate the poisson distributed data using using forward model 
% and poissrnd function
%--------------------------------------------------------------------
  bkgrnd = 1;
  Photon_count = reshape(poissrnd(A*Emission(:)+bkgrnd),nphi,ns);
  SNR = sqrt(norm(A*Emission(:)+bkgrnd)^2/sum(A*Emission(:)+bkgrnd))
  
%------------------------------------------------------------------------
%  Save data in structure array Data.
%------------------------------------------------------------------------
  Data.nx              = n;
  Data.ny              = n;
  Data.Aparams         = Aparams;
  Data.gaussian_stdev  = 0;
  Data.Poiss_bkgrnd    = bkgrnd;
  Data.blankscan       = NaN;       %    blankscan;
  Data.noisy_data      = Photon_count;
  Data.Amult           = 'Amult_PET';
  Data.ctAmult         = 'ctAmult_PET';
  f_temp               = [];
%------------------------------------------------------------------------
%  Display data.
%------------------------------------------------------------------------
  figure(1)
    imagesc(Emission)
    axis('square')
    colormap(1-gray), colorbar
    %title('Object (True Image)')
  figure(2)
    imagesc(Data.noisy_data)
    axis('square')
    colormap(1-gray), colorbar
    %title('Blurred, Noisy Image')
