%Min_poisson.m
%------------------------------------------------------------------------
%  This file implements an optimization routine for solving
%
%       min l(f)+ .5*alpha*f'*L*f    s.t.  f>=0
%
%  where l(f) is a convex function, e.g. the least squares or negative-log
%  of the Poisson likelihood function.
% 
%
%--------------------------------------------------------------------------
 nx = Data.nx;
 ny = Data.ny;
 n  = nx*ny;
    %  Set up and store regularization operator.
  Cost_params.reg_choice = input(' Enter 0 for Tikhonov; 1 for TV; and 2 for Laplacian. ');
  if Cost_params.reg_choice == 2;
     TVflag = 0;
     I = speye(nx,nx);
     D = spdiags([-ones(nx,1),ones(nx,1)],[0,1],nx,nx); D(nx,1)=1;
     L1 = kron(I,D);
     L2 = kron(D,I);
     %if isempty(f_temp)
     %    D1=speye(n^2,n^2);
     %    D2=speye(n^2,n^2);
     %else
     %    D1 = spdiags(1./sqrt((L1*f_temp(:)).^2+0.01),0,nx^2,nx^2);
     %    D2 = spdiags(1./sqrt((L2*f_temp(:)).^2+0.01),0,nx^2,nx^2);
     %end
     %L = L1'*(D1*L1) + L2'*(D2*L2);
     L = L1'*L1 + L2'*L2;
     %L=laplacian(nx,ny);
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
    TVflag = 0;
    Cost_params.reg_matrix = speye(nx*ny);
  end
  
  Cost_params.TVflag            = TVflag;
  Cost_params.nx                = Data.nx;
  Cost_params.ny                = Data.ny;
  Cost_params.data              = Data.noisy_data;
  Cost_params.gaussian_stdev    = Data.gaussian_stdev;
  Cost_params.Poiss_bkgrnd      = Data.Poiss_bkgrnd;
  Cost_params.Aparams           = Data.Aparams;
  Cost_params.cost_fn           = 'poisslhd_fun';
  Cost_params.hess_fn           = 'poisslhd_hess';
  Cost_params.Amult             = Data.Amult;
  Cost_params.ctAmult           = Data.ctAmult;
  Cost_params.precond_fn        = [];
  
  
%------------------------------------------------------------------------
%  Set up and store optimization parameters in structure array Opt_params.
%------------------------------------------------------------------------
  Opt_params.max_iter          = 50;  %input(' Max number of iterations = ');
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
  
%---------------------------------------------------------------------
%  Solve minimization problem.
%---------------------------------------------------------------------
  f_0 = ones(nx,ny);
  method_flag = input('Enter 1 for GCV, 2 for DP, 3 for UPRE, or 4 for no method ');
  if method_flag == 3
      options = optimset('Tolx',1e-8,'Display','Iter');
      lb = input('Enter lower bound for alpha ');
      ub = input('Enter upper bound for alpha ');
      [alpha,fval,exitflag] = fminbnd(@(x) upre_fun(x,f_0,Opt_params,Cost_params,Output_params),lb,ub,options)
  elseif method_flag == 2
      options = optimset('Tolx',1e-8,'Display','Iter'); 
      lb = input('Enter lower bound for alpha ');
      ub = input('Enter upper bound for alpha ');
      alpha = fminbnd(@(x) disc_princ(x,f_0,Opt_params,Cost_params,Output_params),lb,ub,options);
  elseif method_flag == 1
      options = optimset('Tolx',1e-3,'Display','Iter'); 
      lb = input('Enter lower bound for alpha ');
      ub = input('Enter upper bound for alpha ');
      [alpha,fval,exitflag] = fminbnd(@(x) gcv_fun(x,f_0,Opt_params,Cost_params,Output_params),lb,ub,options)
  elseif method_flag == 4
      alpha = input('Enter a value for alpha ');
  end
  Cost_params.reg_param = alpha;
  cpu_t0 = cputime;
  [f_temp,histout] = gpnewton(f_0,Opt_params,Cost_params,Output_params);
  total_cpu_time = cputime - cpu_t0;

%---------------------------------------------------------------------
%  Display results.
%---------------------------------------------------------------------
xtrue = Emission;
poiss_rel_err=norm(f_temp(:)-xtrue(:))/norm(xtrue(:));
 
fprintf('Relative soln error             = %f\n',poiss_rel_err);
fprintf('Total cost function evaluations = %d\n',TOTAL_COST_EVALS);
fprintf('Total grad function evaluations = %d\n',TOTAL_GRAD_EVALS);
fprintf('Total Hess function evaluations = %d\n',TOTAL_HESS_EVALS);
fprintf('Total fast fourier transforms   = %d\n',TOTAL_FFTS);
fprintf('Total preconditioner setups     = %d\n',TOTAL_PRECOND_SETUPS);
fprintf('Total preconditioner evals      = %d\n',TOTAL_PRECOND_EVALS);
fprintf('Total CPU time, in seconds      = %d\n',total_cpu_time);
fprintf('FFT CPU time                    = %d\n',TOTAL_FFT_TIME);
fprintf('Preconditioning time            = %d\n',TOTAL_PRECOND_TIME);

figure
imagesc(f_temp)
axis('square'),colormap(1-gray),colorbar
axis('off')