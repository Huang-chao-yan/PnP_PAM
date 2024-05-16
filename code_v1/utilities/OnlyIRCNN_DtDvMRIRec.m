function [u,v,D,Res] = OnlyIRCNN_DtDvMRIRec(scaleNoi,xi,B,picks,uini,oimg,opts,alpha,rho)
v=0;
D=0;

load('models\modelgray.mat');
if ~exist('opts','var'); opts = []; end

[m,n] = size(B);
F = @(x) fft2(x);
FT = @(x) real(ifft2(x));
% F = @(x) fft2(x)/sqrt(m*n);
% FT = @(x) sqrt(m*n)*ifft2(x,'symmetric');
% Set up the option fields
[mit_learn,mit_inn,mit_out,tol] = setopts(opts);
% Pre-determine Low Pass Filter

% Initialization
u = rand(size(uini));
z=u;


Res = [];
%disp('Initial Frames and Coefficients are Computed');

%% Main Loop
for nstep = 1 : mit_out
    
    uprev = u; 
    erprev=inf;
    % Solve u
    for innerstep = 1 : mit_inn
        
        uinnerprev = u;
        
        u = u-rho*(FT(picks.*(F(u)-B))+(xi+alpha)*(sqrt(2).*u-(z+uprev)./sqrt(2)));
        
        u = min(max(u,0),1);
        
        errinner = norm(u-uinnerprev,'fro')/norm(u,'fro');
        
        if errinner <= tol || errinner>erprev
            %disp(['u Subproblem Step = ' num2str(innerstep), ' error = ' num2str(errinner)]);
            break;
        end
        erprev=errinner;
    end
    if errinner > tol
        %disp(['u Subproblem Step = ' num2str(innerstep), ' error = ' num2str(errinner)]);
    end
    
    % Update Frames and Coefficients
    modelSigma=estimate_noise(u)*255*scaleNoi;
    %disp(modelSigma)
    %imshow(u,[]);title(num2str(norm(u-oimg,'fro')/norm(oimg,'fro')));drawnow;
    [net] = loadmodel(modelSigma,CNNdenoiser);%
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu');

    res = vl_simplenn(net, gpuArray(single(u)),[],[],'conserveMemory',true,'mode','test');
    residual = res(end).x;%
    z=gather(double(u-residual));
    
    err = norm(u-uprev,'fro')/norm(u,'fro');
    
    Res = [Res psnr(u,oimg)];
    
    plot(Res)
    imshow(u)
    drawnow
    if err <= tol
        %disp(['Problem Step = ' num2str(nstep) ' ; Error = ' num2str(norm(u-oimg,'fro')/norm(oimg,'fro')) ' ; error = ' num2str(err)]);
        break;
    end
end

%% Subfunction
function [mit_learn,mit_inn,mit_out,tol,alpha,beta,gamma,rho] = setopts(opts)

% Define Default Option Fields
mit_learn = 30;
mit_inn = 200;
mit_out = 200;
tol = 1e-4;
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