%  GCV.m
%
%  Find the minimizer of the Generalized Cross Validation Functional 
%  GCV(alpha)= n||C^(-1/2)(Af_alpha-(z-gamma)||^2/...
%               trace(I-C^(-1/2)AA_alpha)^2
%  where C = diag(z+sigma^2)
%  f_alpha is the minimizer of the penalized poisson log likelihood
%  functional subject to nonnegativity constraints.
%  
%------------------------------------------------------------------------
%  Set up parameters and data to evaluate cost function and its
%  Hessian. Store information in structure array Cost_params. 
%------------------------------------------------------------------------

  nx = Data.nx;
  ny = Data.ny;
  n = nx*ny;
  
  %  Set up and store regularization operator.
    Cost_params.reg_choice = input(' Enter 0 for Tikhonov; 1 for TV; and 2 for Laplacian. ');

  if Cost_params.reg_choice == 2;
      Tvflag = 0;
     diffusion_flag = input(' Enter 0 for regular Laplacian and 1 for scaling computed from approximate. ');
     if diffusion_flag == 1 
       [Fxf,Fyf]=gradient(f_poisson);
       absGrad = Fxf.^2+Fyf.^2;
       thresh = absGrad.*(absGrad>0.055*max(absGrad(:)));
       Ddiag = max(1./(1+thresh),0.1);%eps^(1/2)/alpha);
       Dmat = spdiags(Ddiag(:),0,n,n);
       L=DiffusionMatrix(nx,ny,Dmat);
     else
       %L=DiffusionMatrix(nx,ny,speye(n,n));
       L=laplacian(nx,ny);
     end
     Cost_params.reg_matrix = L;
  elseif Cost_params.reg_choice == 1  % Total Variation regularization operator.
    %  Set up discretization of first derivative operators.
    TVflag = 1;
    nsq = nx^2;
    Delta_x = 1 / nx;
    Delta_y = Delta_x;
    Delta_xy = 1;%Delta_x*Delta_y;
    D = spdiags([-ones(nx-1,1) ones(nx-1,1)], [0 1], nx-1,nx) / Delta_x;
    I_trunc1 = spdiags(ones(nx-1,1), 0, nx-1,nx);
    I_trunc2 = spdiags(ones(nx-1,1), 1, nx-1,nx);
    Dx1 = kron(D,I_trunc1);
    Dx2 = kron(D,I_trunc2);
    Dy1 = kron(I_trunc1,D);
    Dy2 = kron(I_trunc2,D);
    % Necessaries for cost function evaluation.
    Cost_params.Dx1 = Dx1;
    Cost_params.Dx2 = Dx2;
    Cost_params.Dy1 = Dy1;
    Cost_params.Dy2 = Dy2;
    Cost_params.Delta_xy = Delta_xy;
    Cost_params.beta = 1;
  else              %  Identity regularization operator.
    Cost_params.reg_matrix = speye(nx*ny);
    TVflag = 0;
  end
  Cost_params.TVflag            = TVflag;  
  Cost_params.nx                = nx;
  Cost_params.ny                = ny;
  Cost_params.cost_fn           = 'poisslhd_fun';
  Cost_params.hess_fn           = 'poisslhd_hess';
  Cost_params.data              = Data.noisy_data;
  Cost_params.Aparams           = Data.Aparams;
  Cost_params.gaussian_stdev    = Data.gaussian_stdev;
  Cost_params.Poiss_bkgrnd      = Data.Poiss_bkgrnd;
  Cost_params.Amult             = Data.Amult;
  Cost_params.ctAmult           = Data.ctAmult;
  Cost_params.precond_fn        = [];
    
  

  
%------------------------------------------------------------------------
%  Set up and store optimization parameters in structure array Opt_params.
%------------------------------------------------------------------------

Opt_params.max_iter          = 15;  %input(' Max number of iterations = ');
  Opt_params.step_tol          = 1e-5;  % termination criterion: norm(step)
  Opt_params.grad_tol          = 1e-5;  % termination criterion: rel norm(grad)
  Opt_params.max_gp_iter       = 5;  %input(' Max gradient projection iters = ');
  Opt_params.max_cg_iter       = 30;  %input(' Max conjugate gradient iters = ');
  Opt_params.gp_tol            = 0.1;  % Gradient Proj stopping tolerance.
  p_flag                       = 0; % No preconditioning.
  if p_flag == 1
    Opt_params.cg_tol          = 0.25;   % CG stopping tolerance with precond.
  else
    Opt_params.cg_tol          = 0.1;   % CG stopping tolerance w/o precond.
  end
  Opt_params.linesrch_param1   = 0.01;  % Update parameters for quadratic 
  Opt_params.linesrch_param2   = 0.5;  %   backtracking line search. 
  Opt_params.linesrch_gp       = 'linesrch_gp';  % Line search function.
  Opt_params.linesrch_cg       = 'linesrch_cg';  % Line search function.  
  Output_params.disp_flag      = 1; % 1 to display and 0 otherwise.
  
%------------------------------------------------------------------------
%  Declare and initialize global variables.
%------------------------------------------------------------------------
  
  global TOTAL_COST_EVALS TOTAL_GRAD_EVALS TOTAL_HESS_EVALS TOTAL_FFTS
  global TOTAL_PRECOND_SETUPS TOTAL_PRECOND_EVALS
  global TOTAL_FFT_TIME TOTAL_PRECOND_TIME
  TOTAL_COST_EVALS = 0; 
  TOTAL_GRAD_EVALS = 0; 
  TOTAL_HESS_EVALS = 0; 
  TOTAL_FFTS = 0;
  TOTAL_PRECOND_SETUPS = 0;
  TOTAL_PRECOND_EVALS = 0;
  TOTAL_FFT_TIME = 0;
  TOTAL_PRECOND_TIME = 0;
  
  %--------------------------------------------------------
  % Discrepancy Principle
  %--------------------------------------------------------------


   
%---------------------------------------------------------------------
%  Solve minimization problem.
%---------------------------------------------------------------------
f_0 = ones(nx,ny);
%lhb = input('enter lower-bound for search region ');
%  rhb = input('enter upper-bound for search region ');
  options = optimset('Tolx',1e-8,'Display','Iter');  options = optimset('Tolx',1e-8,'Display','iter');

 [alpha,fval,exitflag] = fminbnd(@(x) gcv_fun(x,f_0,Opt_params,Cost_params,Output_params),-12,-2,options)

 %{
 Cost_params.reg_param=alpha;
    [f,histout] = gpnewton(f_0,Opt_params,Cost_params,Data);
    norm(f(:)-Data.object(:))/norm(Data.object(:))
   % figure(9)
   % imagesc(f);colorbar;;colormap(1-gray)
 %}
