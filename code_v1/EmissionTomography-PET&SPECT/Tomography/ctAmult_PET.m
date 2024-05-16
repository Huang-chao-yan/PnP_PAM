  function y = ctAmult_PET(x,Aparams)

  A    = Aparams.A;
  N    = Aparams.N;
  
  y=reshape(A'*x(:),N,N);
