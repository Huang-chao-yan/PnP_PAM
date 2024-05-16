  function y = Amult_PET(x,Aparams)

  A    = Aparams.A;
  nphi = Aparams.nphi;
  ns   = Aparams.ns;
  
  y=reshape(A*x(:),nphi,ns);
