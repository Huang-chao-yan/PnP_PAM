function [u1,u2,v1,v2,Res1,Res2] = WaveletFrameBiModalityBalanced(f,c,P,Pt,N,B,picks,lambda,kappa,mu1,mu2,frame,Level,wLevel,u1ini,u2ini,oimg1,oimg2,opts)

if ~exist('opts','var'); opts = []; end

F2 = @(x) fft2(x)/N;
F2T = @(x) N*ifft2(x,'symmetric');

[D,R] = GenerateFrameletFilter(frame);
W = @(x) FraDecMultiLevel(x,D,Level); % Frame decomposition
WT = @(x) FraRecMultiLevel(x,R,Level); % Frame reconstruction
nD = length(D);

% Set up the option fields
[mit_inn,mit_out,tol,alpha1,alpha2,gamma,rho1,rho2] = setopts(opts);
% [mit_inn,mit_out,tol,alpha1,alpha2,gamma,rho1,rho2,eta1,eta2] = setopts(opts);

muLevel = getwThresh(lambda,wLevel,Level,D);

% normf = norm(f);
% normB = norm(B,'fro');

Const = Pt*ones(size(f));
ConstImg = reshape(Const,N,N);

% Initialization
u1 = u1ini; u2 = u2ini;
C1 = W(u1); C2 = W(u2);
[v1,v2] = JointHardThreshold(C1,C2,mu1,mu2,muLevel);
Res1 = []; Res2 = [];

%% Main Loop
for nstep = 1 : mit_out
    
    u1prev = u1; u2prev = u2;
    v1prev = v1; v2prev = v2;
    
    % Solve u1
    for innerstep = 1 : mit_inn
        
        u1innerprev = u1;

        u1 = u1-rho1*u1./ConstImg.*(ConstImg-reshape(Pt*(f./(P*u1(:)+c)),N,N)+mu1*(u1-WT(v1))+alpha1*(u1-u1prev));

        u1 = min(max(u1,0),1);
        
        errinner = norm(u1-u1innerprev,'fro')/norm(u1,'fro');
        
        if errinner <= tol
            disp(['u1 Subproblem Step = ' num2str(innerstep), ' error = ' num2str(errinner)]);
            break;
        end
    end
    if errinner > tol
        disp(['u1 Subproblem Step = ' num2str(innerstep), ' error = ' num2str(errinner)]);
    end
    
    % Solve u2
    for innerstep2 = 1 : mit_inn
        
        u2innerprev = u2;
        
        u2 = u2-rho2*(kappa*F2T(picks.*(F2(u2)-B))+mu2*(u2-WT(v2))+alpha2*(u2-u2prev));

        u2 = min(max(u2,0),1);
        
        errinner2 = norm(u2-u2innerprev,'fro')/norm(u2,'fro');
        if errinner2 <= tol
            disp(['u2 Subproblem Step = ' num2str(innerstep2), ' error = ' num2str(errinner2)]);
            break;
        end
    end
    if errinner2 > tol
        disp(['u2 Subproblem Step = ' num2str(innerstep2), ' error = ' num2str(errinner2)]);
    end
    
    % Update v = (v1,v2)
    C1 = W(u1); C2 = W(u2);
    v1temp = CoeffOper('+',CoeffOper('*',C1,mu1/(mu1+gamma)),CoeffOper('*',v1prev,gamma/(mu1+gamma)));
    v2temp = CoeffOper('+',CoeffOper('*',C2,mu2/(mu2+gamma)),CoeffOper('*',v2prev,gamma/(mu2+gamma)));
    
    for ki = 1 : Level
        v1temp{ki}{1,1} = C1{ki}{1,1};
        v2temp{ki}{1,1} = C2{ki}{1,1};
    end

    [v1,v2] = JointHardThreshold(v1temp,v2temp,mu1+gamma,mu2+gamma,CoeffOper('*',muLevel,2));
    
    err = norm(u1-u1prev,'fro')/norm(u1,'fro') + norm(u2-u2prev,'fro')/norm(u2,'fro');
    
    Res1 = [Res1 norm(u1-oimg1,'fro')/norm(oimg1,'fro')];
    Res2 = [Res2 norm(u2-oimg2,'fro')/norm(oimg2,'fro')];
    
    if mod(nstep,20) == 0
        disp(['Problem Step = ' num2str(nstep) ' ; Error1 = ' num2str(norm(u1-oimg1,'fro')/norm(oimg1,'fro')) ' ; Error2 = ' num2str(norm(u2-oimg2,'fro')/norm(oimg2,'fro')) ' ; error = ' num2str(err)]);
    end
    
    if err <= tol
        disp(['Problem Step = ' num2str(nstep) ' ; Error1 = ' num2str(norm(u1-oimg1,'fro')/norm(oimg1,'fro')) ' ; Error2 = ' num2str(norm(u2-oimg2,'fro')/norm(oimg2,'fro')) ' ; error = ' num2str(err)]);
        break
    end
    
end

%% Subfunction
function [mit_inn,mit_out,tol,alpha1,alpha2,gamma,rho1,rho2] = setopts(opts)

% Define Default Option Fields
mit_inn = 200;
mit_out = 200;
tol = 1e-4;
alpha1 = 1;
alpha2 = 1;
gamma = 0.005;
rho1 = 1;
rho2 = 1;

% Change to Specified Option Fields if Exist
if ~isempty(opts);
    if ~isa(opts,'struct'); error('WaveletFrameBiModalityBalanced : opts not a struct'); end
    if isfield(opts,'mit_inn'); mit_inn = opts.mit_inn; end
    if isfield(opts,'mit_out'); mit_out = opts.mit_out; end
    if isfield(opts,'tol'); tol = opts.tol; end
    if isfield(opts,'alpha1'); alpha1 = opts.alpha1; end
    if isfield(opts,'alpha2'); alpha2 = opts.alpha2; end
    if isfield(opts,'gamma'); gamma = opts.gamma; end
    if isfield(opts,'rho1'); rho1 = opts.rho1; end
    if isfield(opts,'rho2'); rho2 = opts.rho2; end
end
return;

% function [mit_inn,mit_out,tol,alpha1,alpha2,gamma,rho1,rho2,eta1,eta2] = setopts(opts)
% 
% % Define Default Option Fields
% mit_inn = 200;
% mit_out = 200;
% tol = 1e-4;
% alpha1 = 1;
% alpha2 = 1;
% gamma = 0.005;
% rho1 = 1;
% rho2 = 1;
% eta1 = 0.005;
% eta2 = 0.005;
% 
% % Change to Specified Option Fields if Exist
% if ~isempty(opts);
%     if ~isa(opts,'struct'); error('WaveletFrameBiModalityBalanced : opts not a struct'); end
%     if isfield(opts,'mit_inn'); mit_inn = opts.mit_inn; end
%     if isfield(opts,'mit_out'); mit_out = opts.mit_out; end
%     if isfield(opts,'tol'); tol = opts.tol; end
%     if isfield(opts,'alpha1'); alpha1 = opts.alpha1; end
%     if isfield(opts,'alpha2'); alpha2 = opts.alpha2; end
%     if isfield(opts,'gamma'); gamma = opts.gamma; end
%     if isfield(opts,'rho1'); rho1 = opts.rho1; end
%     if isfield(opts,'rho2'); rho2 = opts.rho2; end
%     if isfield(opts,'eta1'); eta1 = opts.eta1; end
%     if isfield(opts,'eta2'); eta2 = opts.eta2; end
% end
% return;

function [D,R] = GenerateFrameletFilter(frame)

if (nargin < 1) || (isempty(frame)); frame = 1; end
if (frame~=0) && (frame~=1) && (frame~=3)
    error('Input variable frame must be 0, 1, or 3');
end
if frame==0          %Haar Wavelet
    D{1}=[0 1 1]/2;
    D{2}=[0 1 -1]/2;
%     D{3}='rr';
    D{3}='cc';
    R{1}=[1 1 0]/2;
    R{2}=[-1 1 0]/2;
%     R{3}='rr';
    R{3}='cc';
elseif frame==1      %Piecewise Linear Framelet
    D{1}=[1 2 1]/4;
    D{2}=[1 0 -1]/4*sqrt(2);
    D{3}=[-1 2 -1]/4;
%     D{4}='rrr';
    D{4}='ccc';
    R{1}=[1 2 1]/4;
    R{2}=[-1 0 1]/4*sqrt(2);
    R{3}=[-1 2 -1]/4;
%     R{4}='rrr';
    R{4}='ccc';
elseif frame==3      %Piecewise Cubic Framelet
    D{1}=[1 4 6 4 1]/16;
    D{2}=[1 2 0 -2 -1]/8;
    D{3}=[-1 0 2 0 -1]/16*sqrt(6);
    D{4}=[-1 2 0 -2 1]/8;
    D{5}=[1 -4 6 -4 1]/16;
    D{6}='ccccc';
%     D{6}='rrrrr';
    R{1}=[1 4 6 4 1]/16;
    R{2}=[-1 -2 0 2 1]/8;
    R{3}=[-1 0 2 0 -1]/16*sqrt(6);
    R{4}=[1 -2 0 2 -1]/8;
    R{5}=[1 -4 6 -4 1]/16;
    R{6}='ccccc';
%     R{6}='rrrrr';
end
return;