% function R = RegluarizationMatrix(j,n,L1left,L1right,L2top,L2bottom,theta)
%
% The program computes the reglarization matrix with a given variance
% vector. The Dirichlet boundary conditions are rotated from iteration 
% to iteration to avoid bias   
% Input:   j  -  integer, number of the iteration round
%          n  -  integer, number of pixels per edge
%          L1left (L1right) - n^2 x n^2 sparse matrix, horizontal derivative 
%          with Dirichlet boundary condition at the left (right) edge
%          L2top (L2bottom) - n^2 x n^2 sparse matrix, vertical derivative 
%          with Dirichlet boundary condition at the top (bottom) edge
%          theta  -  n^2 vector, pixelwise prior variances
%
% Output:  L = L1'*D*L1 + L2'*D*L2;
% 
% Calls to: None
%
function L = RegluarizationMatrix(j,n,L1,L2)%L1left,L1right,L2top,L2bottom,theta)
% Rotating the boundary conditions
%if mod(j,4) == 1
%   L1 = L1left; L2 = L2top;
%elseif mod(j,4) == 2
%   L1 = L1right; L2 = L2top;
%elseif mod(j,4) == 3
%   L1 = L1right; L2 = L2bottom;
%elseif mod(j,4) == 0
%   L1 = L1left; L2 = L2bottom;
%end
D = spdiags(1./theta,0,n^2,n^2);
L = L1'*D*L1 + L2'*D*L2;
