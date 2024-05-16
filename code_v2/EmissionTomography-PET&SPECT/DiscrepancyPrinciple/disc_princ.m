function a = disc_princ(alpha,f_0,Opt_params,Cost_params,Output_params)

%
%  Evaluate (r(alpha)/n-1)^2 where
%  r(alpha) = ||C^(-1/2)(Tf_alpha-(d-b))||^2
%

Cost_params.reg_param=alpha;
nx      = Cost_params.nx;
ny      = Cost_params.ny;
d       = Cost_params.data;
b       = Cost_params.Poiss_bkgrnd;
sig     = Cost_params.gaussian_stdev;
Amult   = Cost_params.Amult;
ctAmult = Cost_params.ctAmult;
Aparams = Cost_params.Aparams;

n = nx*ny;
[f] = gpnewton(f_0,Opt_params,Cost_params,Output_params);
Kf = feval(Amult,f,Aparams);
W = 1./(Kf + b + sig^2);
%W  = 1./(d+b+sig^2);
resid_vec = Kf - (d-b);
Wr = W .* resid_vec;
F_alpha = resid_vec(:)'*Wr(:);
a = (F_alpha/n-1)^2;