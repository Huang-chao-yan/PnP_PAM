  function [J,Params,g] = poisslhd_fun(f,Params)
  
%  Evaluate penalized Poisson log likelihood cost functional
%      J(f) = sum[(A*f+c) -(d+sig^2).*log(A*f+c)] + alpha/2*f'*L*f.
%  Here d is a realization from a statistical distribution
%      d ~ Poisson(A*f) + Poisson(b) + normal(0,sig^2).
%  Ahe constant c = b + sig^2. 
%  The first term in J(f) is the log likelihood functional for the
%  distributional approximation
%      d_i + sig^2 ~ Poisson([A*f]_i) + Poisson(b) + Poisson(sig^2).
%    The gradient of J is given by 
%      g(f) = A'*[(A*f+b-d)./(A*f+c)] + alpha*L*f.

  %  Declare global variables and initialize parameters and vectors.

  global TOTAL_COST_EVALS TOTAL_GRAD_EVALS
  TOTAL_COST_EVALS = TOTAL_COST_EVALS + 1;

  Amult   = Params.Amult;
  ctAmult = Params.ctAmult;
  Aparams = Params.Aparams;
  d       = Params.data;
  b       = Params.Poiss_bkgrnd;
  sig     = Params.gaussian_stdev;
  nx      = Params.nx;
  ny      = Params.ny;
  alpha   = Params.reg_param;
  TVflag  = Params.TVflag;
  
  %------------  Compute J(f).  -----------------%
  c = b + sig^2;
  dps = d + sig^2;
  Afpc = feval(Amult,f,Aparams) + c;
  Params.Afpc = Afpc;
  Jfit = sum(Afpc(:) - dps(:).*log(Afpc(:)));
  
  if TVflag == 0
      L  = Params.reg_matrix;
      Lf = L*f(:); 
      Jreg = .5*alpha*(f(:)'*Lf);
  elseif TVflag == 1
    Dx1      = Params.Dx1;
    Dx2      = Params.Dx2;
    Dy1      = Params.Dy1;
    Dy2      = Params.Dy2;
    Delta_xy = Params.Delta_xy;
    beta     = Params.beta;
    Du_sq1   = (Dx1*f(:)).^2 + (Dy1*f(:)).^2;
    Du_sq2   = (Dx2*f(:)).^2 + (Dy2*f(:)).^2;
    Jreg1    = .5 * Delta_xy * sum(psi_fun(Du_sq1,beta));
    Jreg2    = .5 * Delta_xy * sum(psi_fun(Du_sq2,beta));
    Jreg     = alpha * (Jreg1 + Jreg2 ) / 2;
  end

  J = Jfit + Jreg;
  
  
  %------------  Compute grad J(f).  -----------------%
  q = (Afpc - dps) ./ Afpc;
  gfit = feval(ctAmult,q,Aparams);
  if TVflag == 0
      g = gfit + alpha*reshape(Lf,nx,ny);
  elseif TVflag == 1
    psi_prime1 = psi_prime(Du_sq1, beta);
    psi_prime2 = psi_prime(Du_sq2, beta);
    Dpsi_prime1 = spdiags(psi_prime1, 0, (nx-1)^2,(nx-1)^2);
    Dpsi_prime2 = spdiags(psi_prime2, 0, (nx-1)^2,(nx-1)^2);
    L1 = Dx1' * Dpsi_prime1 * Dx1 + Dy1' * Dpsi_prime1 * Dy1;
    L2 = Dx2' * Dpsi_prime2 * Dx2 + Dy2' * Dpsi_prime2 * Dy2;
    L = (L1 + L2) * Delta_xy / 2;
    Params.reg_matrix = L;
    Lf = L*f(:);
    g = gfit + alpha*reshape(Lf,nx,ny);
  end

      
  TOTAL_GRAD_EVALS = TOTAL_GRAD_EVALS + 1;
  
