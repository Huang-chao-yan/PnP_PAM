function a = upre_fun(alpha,f_0,Opt_params,Cost_params,Output_params)

%  This function evaluates the UPRE function:
%  UPRE(alpha)=(1/n)||r(alpha)||^2+(2/n)trace(C^(-1/2)AA_alpha)-1
%  Note:  trace(C^(-1/2)AA_alpha) is computed using randomized trace
%         estimation


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

Walpha = 1./(Kf+b+sig^2);
Cost_params.Walpha = Walpha;
resid_vec = feval(Amult,f,Aparams) - (d-b);
Wr = Walpha .* resid_vec;
F_alpha = resid_vec(:)'*Wr(:);

F_alpha=.5*F_alpha;
v_vec = rand(nx,ny);
v_vec = (v_vec>=.5)-(v_vec<.5);
if Cost_params.TVflag == 1
    Dx1      = Cost_params.Dx1;
    Dx2      = Cost_params.Dx2;
    Dy1      = Cost_params.Dy1;
    Dy2      = Cost_params.Dy2;
    Delta_xy = Cost_params.Delta_xy;
    beta     = Cost_params.beta;
    Du_sq1   = (Dx1*f(:)).^2 + (Dy1*f(:)).^2;
    Du_sq2   = (Dx2*f(:)).^2 + (Dy2*f(:)).^2;
    psi_prime1 = psi_prime(Du_sq1, beta);
    psi_prime2 = psi_prime(Du_sq2, beta);
    Dpsi_prime1 = spdiags(psi_prime1, 0, (nx-1)^2,(nx-1)^2);
    Dpsi_prime2 = spdiags(psi_prime2, 0, (nx-1)^2,(nx-1)^2);
    L1 = Dx1' * Dpsi_prime1 * Dx1 + Dy1' * Dpsi_prime1 * Dy1;
    L2 = Dx2' * Dpsi_prime2 * Dx2 + Dy2' * Dpsi_prime2 * Dy2;
    L = (L1 + L2) * Delta_xy / 2+getLprime(f,Cost_params);;
    Cost_params.reg_matrix = L;
else
    L = Cost_params.reg_matrix;
end
D_alpha=(f>0);

Cost_params.D_alpha = D_alpha;
Cost_params.max_cg_iter = 100;%      Maximimum number of CG iterations.
Cost_params.cg_step_tol = 1e-4; %      Stop CG when ||x_k+1 - x_k|| < step_tol.
Cost_params.grad_tol_factor = 1e-4;%   Stop CG when ||g_k|| < grad_tol_factor*||g_0||.
Cost_params.cg_io_flag = 0;%       Output CG info if ioflag = 1.
Cost_params.cg_figure_no = 10;%     Figure number for CG output.
%keyboard
AAtWv = feval(ctAmult,(Walpha.^(1/2)).*v_vec,Aparams);
bb = D_alpha.*AAtWv;
w=cg_paramselect(ones(n,1),bb(:),Cost_params,'cg_Amult_PET');
w = reshape(w,nx,ny);
Aw = feval(Amult,w,Aparams);
CAw = (Walpha.^(1/2)).*Aw;

a = F_alpha/n+v_vec(:)'*CAw(:)/n-(1/2);
