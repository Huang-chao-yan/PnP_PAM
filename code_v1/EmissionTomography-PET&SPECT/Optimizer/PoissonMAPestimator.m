function[f_poisson,histout] = PoissonMAPestimator(f_0,Data,RegMat)

%------------------------------------------------------------------------
%  This function implements an optimization routine for solving
%
%       min l(f)+ f'*L*f    s.t.  f>=0
%
%  where l(f) is a convex function, e.g. the least squares or negative-log
%  of the Poisson likelihood function.
% 
%  Inputs:
%  f_0    = initial guess for optimizer.
%  Data   = structure array containing necessary info from data generation.
%  RegMat = matrix from regularization term
%
%  Outputs:
%  f_poisson = nonnegative reconstruction
%  histout    = iteration history
%
%  First, enter paramters needed for the evaluation of the cost function.
%------------------------------------------------------------------------
  Cost_params.TVflag            = 0;
  Cost_params.reg_matrix        = RegMat;
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
  Cost_params.reg_param         = 1;
  
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
  
  cpu_t0 = cputime;
  [f_poisson,histout] = gpnewton(f_0,Opt_params,Cost_params,Output_params);
  total_cpu_time = cputime - cpu_t0;
    
%---------------------------------------------------------------------
%  Display results.
%---------------------------------------------------------------------
