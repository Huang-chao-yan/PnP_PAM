function a = errmin_fun(alpha,f_0,Opt_params,Cost_params,Output_params,Emission)

%
%  Evaluate the Generalized Cross Validation Functional
%  GCV(alpha)= n||C^(-1/2)(Af_alpha-(z-gamma)||^2/...
%               trace(I-C^(-1/2)AA_alpha)^2
%  where
%  A_alpha = (D_a(T'C^(-1)T+alphaL)D_a)^(dagger)D_aT'C^(-1/2)
%  Note that randomized trace estimation is used.

% Extract needed parameters
    Cost_params.reg_param=alpha;
    Aparams = Cost_params.Aparams;
    d     = Cost_params.data;
    b     = Cost_params.Poiss_bkgrnd;
    sig   = Cost_params.gaussian_stdev;
    Amult = Cost_params.Amult;
    ctAmult = Cost_params.ctAmult;

% Solve for f_alpha
[f] = gpnewton(f_0,Opt_params,Cost_params,Output_params);

a= norm(f(:)-Emission(:))/norm(Emission(:));
