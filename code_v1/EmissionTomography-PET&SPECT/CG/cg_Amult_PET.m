  function y = cg_Amult_PET(x,params)

   Aparams = params.Aparams;
   D_alpha = params.D_alpha;
   L = params.reg_matrix;
   alpha = params.reg_param;
   W = params.Walpha;
   nx = params.nx;
   ny = params.ny;
   x = reshape(x,nx,ny);
  
   A  = Aparams.A;
   Dx = D_alpha.*x;
   aLDx = reshape(alpha*L*Dx(:),nx,ny);
   DaLDx = D_alpha.*aLDx;
  
   ADx=reshape(A*x(:),nx,ny);
   WADx=W.*ADx;
   AtWADx = reshape(A'*WADx(:),nx,ny);
   DAtWADx = D_alpha.*AtWADx;
   y = DAtWADx(:)+DaLDx(:);
