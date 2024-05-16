% Program DeblurDriver
% Modular version of the Iterative Adaptive Sequential deblurring algorithm
%--------------------------------------------------------------
% First generate data. For an example, see SetupSatellite.m or
% SetupSinogram.m. 
%
% Next generate first derivative matrices for regularization.
%--------------------------------------------------------------
% Calculating the gradient matrix. To avoid biasing due to the one-sided
% Dirichlet boundary condition, the boundary conditions are rotated from
% iteration to iteration. Therefore, four finite difference matrices are
% calculated, corresponding to Dirichlet data at each side of the image.
n = Data.nx;
I = speye(n,n);
D = spdiags([-ones(n,1),ones(n,1)],[0,1],n,n); D(n,1)=1;
L1 = kron(I,D);
L2 = kron(D,I);

%--------------------------------------------------------------
% Iteration.
%--------------------------------------------------------------
%alpha0 = 2; theta0 = 1/(alpha0*alpha); % inv-Gamma hyper-priors
theta1 = alpha*ones(n^2,1);  % Initial guess for prior parameter  
theta2 = theta1;
x0 = ones(n,n);              % Initial guess for reconstruction
niter = 10;                   % Number of sequential updates
output_flag = 0;
errorvec = norm(x0(:)-Emission(:))/norm(Emission(:));
tic
for j = 1:niter
    disp(['Iteration = ' num2str(j)]);
    
    % Calculate the regularization matrix
    D1 = spdiags(theta1,0,n^2,n^2);
    D2 = spdiags(theta2,0,n^2,n^2);
    L = L1'*D1*L1 + L2'*D2*L2;
    
    % Update the image: compute the MAP estimator. 
    [X,histout] = PoissonMAPestimator(x0,Data,L);
    errorvec = [errorvec norm(X(:)-Emission(:))/norm(Emission(:))];
    
    % Update the variance vector from inverse-Gamma hyper-prior
    %eta = 0.5*(alpha0 - 2);
    %theta = theta0*(eta + sqrt((1/(2*theta0))*((L1*X(:)).^2 + (L2*X(:)).^2)+eta^2+0.001));
    theta1 = sqrt(alpha)./sqrt((L1*X(:)).^2+.01);
    theta2 = sqrt(alpha)./sqrt((L2*X(:)).^2+.01);
    
    % Plot the current estimate
    if output_flag == 1 | j==niter
        figure(5)
          imagesc(reshape(X,n,n))
          axis('square'),colormap(1-gray),colorbar
          %axis('off')
          text(10,240,['Iteration ' num2str(j)])
          drawnow
        figure(6)
          imagesc(reshape(sqrt((L1*X(:)).^2+(L2*X(:)).^2),n,n))
          axis('square'),colormap(1-gray),colorbar
          %axis('off')
          text(10,240,['Iteration ' num2str(j)])
          drawnow
    end
    fprintf('relative error = %0.4f\n',norm(X(:)-xtrue(:))/norm(xtrue(:)))
end
toc
