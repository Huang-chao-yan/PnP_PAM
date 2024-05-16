function [u,v,D,Res] = OnlyIRCNN_DtDvPETRec(scaleNoi,xi,f,c,P,Pt,N,uini,oimg,opts,alpha,rho)
load('models\modelgray.mat');
mu=0;
v=1;
D=1;
addpath(genpath('.'));

if ~exist('opts','var'); opts = []; end

% Set up the option fields
[mit_learn,mit_inn,mit_out,tol] = setopts(opts);

Const = Pt*ones(size(f));
ConstImg = reshape(Const,N,N);

% Pre-determine Low Pass Filter


% Initialization
u = uini;
u=rand(size(u));
z=u;
minv=inf;
minu=0;
Res = [];
%disp('Initial Frames and Coefficients are Computed');

%% Main Loop
for nstep = 1 : mit_out
   
    uprev = u; 
    
    % Solve u
    preverr=inf;
    for innerstep = 1 : mit_inn
        
        uinnerprev = u;
        u = u-rho*u./ConstImg.*(ConstImg-reshape(Pt*(f./(P*u(:)+c)),N,N)+(xi+alpha)*(sqrt(2).*u-(z+uprev)./sqrt(2)));        
        u = min(max(u,0),1);
        
        errinner = norm(u-uinnerprev,'fro')/norm(u,'fro');
        
        if errinner <= tol || errinner>preverr
            break;
        end
        preverr=errinner;
    end
    
    
    % Update Frames and Coefficients
    modelSigma=estimate_noise(u)*255*scaleNoi;

    [net] = loadmodel(modelSigma,CNNdenoiser);%
    net = vl_simplenn_tidy(net);
    net = vl_simplenn_move(net, 'gpu');
    % Update Frames and Coefficients
    res = vl_simplenn(net, gpuArray(single(u)),[],[],'conserveMemory',true,'mode','test');
    residual = res(end).x;%
    z=gather(double(u-residual));

    err = norm(u-uprev,'fro')/norm(u,'fro');
    
    Res = [Res psnr(u,oimg)];
   imshow(u)
    drawnow

    if err <= tol
        break;
    end
end
%% Subfunction
function [mit_learn,mit_inn,mit_out,tol] = setopts(opts)

% Define Default Option Fields
mit_learn = 30;
mit_inn = 100;
mit_out = 100;
tol = 5e-4;
alpha = 1;
beta = 0.005;
gamma = 0.005;
rho = 1;

% Change to Specified Option Fields if Exist
if ~isempty(opts);
    if ~isa(opts,'struct'); error('DtDvPETRec : opts not a struct'); end
    if isfield(opts,'mit_inn'); mit_inn = opts.mit_inn; end
    if isfield(opts,'mit_out'); mit_out = opts.mit_out; end
    if isfield(opts,'tol'); tol = opts.tol; end
    if isfield(opts,'alpha'); alpha = opts.alpha; end
    if isfield(opts,'beta'); beta = opts.beta; end
    if isfield(opts,'gamma'); gamma = opts.gamma; end
    if isfield(opts,'rho'); rho = opts.rho; end
end
return;
%%
