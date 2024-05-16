function [u1,u2,Res1,Res2] = WaveletFrameBiModality(f,c,P,Pt,N,B,picks,lambda,kappa,frame,Level,wLevel,oimg1,oimg2,opts)

if ~exist('opts','var'); opts = []; end

F2 = @(x) fft2(x)/N;
F2T = @(x) N*ifft2(x,'symmetric');

[D,R] = GenerateFrameletFilter(frame);
W = @(x) FraDecMultiLevel(x,D,Level); % Frame decomposition
WT = @(x) FraRecMultiLevel(x,R,Level); % Frame reconstruction

% Set up the option fields
[mit_inn,mit_out,tol,beta,rho,u1_ini,u2_ini] = setopts(opts);

Const = Pt*ones(size(f));
ConstImg = reshape(Const,N,N);

% Compute the weighted thresholding parameters.
muLevel = getwThresh(lambda/beta,wLevel,Level,D);

% Initialization
if isempty(u1_ini); u1 = ones(N,N); else u1 = u1_ini; clear u1_ini; end
if isempty(u2_ini); u2 = zeros(N,N); else u2 = u2_ini; clear u2_ini; end
d1 = W(zeros(N,N)); b1 = d1;
d2 = W(zeros(N,N)); b2 = d2;
nD = length(D);
Res1 = []; Res2 = [];

%% Main Loop
for nstep = 1 : mit_out
    
    % Solve u1
    C1 = CoeffOper('-',d1,b1);
    for innerstep = 1 : mit_inn
        
        u1innerprev = u1;
        % EM Step
        u1temp = u1innerprev.*reshape(Pt*(f./(P*u1(:)+c))./Const,N,N);
        
        % Remaining Step
        u1 = (u1temp+beta*u1innerprev./ConstImg.*WT(C1))./(1+beta*u1innerprev./ConstImg);
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
    
%     u1 = min(max(u1,0),ub1)£»
    
    % Solve u2
    C2 = CoeffOper('-',d2,b2);
    for innerstep2 = 1 : mit_inn
        
        u2innerprev = u2;
        
        u2 = u2 - rho*(kappa*F2T(picks.*(F2(u2)-B))+beta*(u2-WT(C2)));
        
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
    
    
    
    % Solve d = (d1,d2)
    C1 = W(u1);
    C2 = W(u2);
    C1pb1 = CoeffOper('+',C1,b1);
    C2pb2 = CoeffOper('+',C2,b2);
    [d1,d2] = JointIsoThresh(C1pb1,C2pb2,muLevel);
    
    err1 = 0;
    err2 = 0;
    for ki = 1 : Level
        for ji = 1 : nD-1
            for jj = 1 : nD-1
                if ((ji~=1)||(jj~=1))||(ki==Level)
                    deltab1 = C1{ki}{ji,jj}-d1{ki}{ji,jj};
                    deltab2 = C2{ki}{ji,jj}-d2{ki}{ji,jj};
                    err1 = err1 + norm(deltab1,'fro')^2;
                    err2 = err2 + norm(deltab2,'fro')^2;
                    b1{ki}{ji,jj} = b1{ki}{ji,jj} + deltab1;
                    b2{ki}{ji,jj} = b2{ki}{ji,jj} + deltab2;
                end
            end
        end
    end
    
    err = sqrt(err1)/cellnorm(C1)+sqrt(err2)/cellnorm(C2);
    
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
function [mit_inn,mit_out,tol,beta,rho,u1_ini,u2_ini] = setopts(opts)

% Define Default Option Fields
mit_inn = 200;
mit_out = 200;
tol = 1e-4;
beta = 0.05;
rho = 1;
u1_ini = [];
u2_ini = [];

% Change to Specified Option Fields if Exist
if ~isempty(opts);
    if ~isa(opts,'struct'); error('WaveletFrameBiModality : opts not a struct'); end
    if isfield(opts,'mit_inn'); mit_inn = opts.mit_inn; end
    if isfield(opts,'mit_out'); mit_out = opts.mit_out; end
    if isfield(opts,'tol'); tol = opts.tol; end
    if isfield(opts,'beta'); beta = opts.beta; end
    if isfield(opts,'rho'); rho = opts.rho; end
    if isfield(opts,'u1_ini'); u1_ini = opts.u1_ini; end
    if isfield(opts,'u2_ini'); u2_ini = opts.u2_ini; end
end
return;

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