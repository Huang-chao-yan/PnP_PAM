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
% From Erkki & Daniela
% I = [1:n^2];
% I = reshape(I,n,n);
% % Horizontal derivative, Diriclet boundary value left
% rows = I(:,2:n); rows = rows(:);
% cols = I(:,1:n-1); cols = cols(:);
% vals = ones(n*(n-1),1);
% L1left = speye(n^2) - sparse(rows,cols,vals,n^2,n^2); 
% % Horizontal derivative, Diriclet boundary value right
% rows = I(:,1:n-1); rows = rows(:);
% cols = I(:,2:n); cols = cols(:);
% vals = ones(n*(n-1),1);
% L1right = speye(n^2) - sparse(rows,cols,vals,n^2,n^2); 
% % Vertical derivative, Dirichlet boundary value top
% rows = I(1:n-1,:); rows = rows(:);
% cols = I(2:n,:); cols = cols(:);
% vals = ones(n*(n-1),1);
% L2top = speye(n^2) - sparse(rows,cols,vals,n^2,n^2);
% % Vertical derivative, Dirichlet boundary value bottom
% rows = I(2:n,:); rows = rows(:);
% cols = I(1:n-1,:); cols = cols(:);
% vals = ones(n*(n-1),1);
% L2bottom = speye(n^2) - sparse(rows,cols,vals,n^2,n^2);

%-------------------------------------------------------------------
% Select the hypermodel. Here the choices are Gamma and InvGamma
%-------------------------------------------------------------------
hypermodel = 'Gamma';
%hypermodel = 'InvGamma';
% Setting the hyperparameters
if strcmp(hypermodel,'Gamma')
    alpha0 = 2.01;
    theta0 = 1/(alpha0*alpha);
elseif strcmp(hypermodel,'InvGamma')
   % Inverse Gamma hypermodel
   theta0 = 300;
   %theta0 = 1;
   alpha0 = 1/theta0;
end

%--------------------------------------------------------------
% Iteration.
%--------------------------------------------------------------
theta = theta0*ones(n^2,1);  % Initial guess for prior parameter  
x0 = ones(n,n);              % Initial guess for reconstruction
niter = 6;                   % Number of sequential updates
output_flag = 0;
errorvec = norm(x0(:)-Emission(:))/norm(Emission(:));
tic
for j = 1:niter
    disp(['Iteration = ' num2str(j)]);
    
    % Calculate the regularization matrix
    %L = RegluarizationMatrix(j,n,L1,L2,theta);%L1left,L1right,L2top,L2bottom,theta);
    D = spdiags(1./theta,0,n^2,n^2);
    L = L1'*D*L1 + L2'*D*L2;
    
    % Update the image: compute the MAP estimator. 
    [X,histout] = PoissonMAPestimator(x0,Data,L);
    errorvec = [errorvec norm(X(:)-Emission(:))/norm(Emission(:))];
    
    % Update the variance vector
    theta = UpdateVariance(j,n,hypermodel,theta0,alpha0,L1,L2,X(:));
    
    % Plot the current estimate
    if output_flag == 1 | j==niter
        figure(5)
          imagesc(reshape(X,n,n))
          axis('square'),colormap(1-gray),colorbar
          %axis('off')
          text(10,240,['Iteration ' num2str(j)])
          drawnow
        figure(6)
          imagesc(reshape(log10(theta),n,n))
          axis('square'),colormap(1-gray),colorbar
          %axis('off')
          text(10,240,['Iteration ' num2str(j)])
          drawnow
    end
    fprintf('relative error = %0.4f\n',norm(X(:)-xtrue(:))/norm(xtrue(:)))
end
toc