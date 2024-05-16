% Program DeblurDriver
% Modular version of the Iterative Adaptive Sequential deblurring algorithm
%--------------------------------------------------------------
% First generate data. For an example, see SetupSinogram.m. 
% Then begin the iteration. 
%--------------------------------------------------------------
x0 = ones(nx,nx);              % Initial guess for reconstruction
% Calculate the regularization matrix
Wdiag = 1./sqrt((Dh*f_temp(:)).^2+(Dv*f_temp(:)).^2+beta);
W = spdiags(Wdiag,[0],nx^2,nx^2);
L = Dv'*(W*Dv)+Dh'*(W*Dh);
% Update the image: compute the MAP estimator. 
[X,histout] = PoissonMAPestimator(x0,Data,alpha*L);
errorvec = [errorvec norm(X(:)-Emission(:))/norm(Emission(:))];
    
figure(5)
  imagesc(X)
  axis('square'),colormap(1-gray),colorbar
  %axis('off')
  text(10,240,['Iteration ' num2str(j)])
  drawnow
figure(6)
  imagesc(reshape(log10(Wdiag),nx,nx))
  axis('square'),colormap(gray),colorbar
  %axis('off')
  text(10,240,['Iteration ' num2str(j)])
  drawnow
fprintf('relative error = %0.4f\n',norm(X(:)-xtrue(:))/norm(xtrue(:)))
