function a = disc_princReconstruct(x,x0,n,L1left,L1right,L2top,L2bottom,Data)

%
%  Evaluate (r(alpha)/n-1)^2 where
%  r(alpha) = ||C^(-1/2)(Tf_alpha-(d-b))||^2
%
   b = Data.Poiss_bkgrnd;
   theta = x*ones(n^2,1); 
   d = Data.noisy_data;
   j = 1;
   

   % Calculate the regularization matrix
    L = RegluarizationMatrix(j,n,L1left,L1right,L2top,L2bottom,theta);
    
    % Update the image: compute the MAP estimator. 
    [X,histout] = PoissonMAPestimator(x0,Data,L);
    %errorvec = [norm(X-Emission)/norm(Emission)];
    
    Kf = feval(Data.Amult,X,Data.Aparams);
    W = 1./(Kf + b);
    %W  = 1./(d+b+sig^2);
    resid_vec = Kf - (d-b);
    Wr = W .* resid_vec;
    F_alpha = resid_vec(:)'*Wr(:);
    a = (F_alpha/n^2-1)^2;