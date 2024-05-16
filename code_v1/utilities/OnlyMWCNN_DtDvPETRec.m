function [u,v,D,Res] = OnlyMWCNN_DtDvPETRec(scaleNoi,xi,f,c,P,Pt,N,uini,oimg,opts,alpha,rho)
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
% u=rand(size(u));
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
            %disp(['u Subproblem Step = ' num2str(innerstep), ' error = ' num2str(errinner)]);
            break;
        end
        preverr=errinner;
    end
    if errinner > tol
        %disp(['u Subproblem Step = ' num2str(innerstep), ' error = ' num2str(errinner)]);
    end
    
    % Update Frames and Coefficients
    modelSigma=estimate_noise(u)*255*scaleNoi;
    %disp(modelSigma)
    %imshow(u,[]);title(num2str(norm(u-oimg,'fro')/norm(oimg,'fro')));drawnow;
    m=max(min(ceil(modelSigma/2)*2,50),2);
    load(fullfile('models\',num2str(m)));
    
    net = dagnn.DagNN.loadobj(net) ;
    net.removeLayer('objective') ;
    out_idx = net.getVarIndex('prediction') ;
    net.vars(net.getVarIndex('prediction')).precious = 1 ;
    net.mode = 'test';
    
    net.move('gpu');
    z=Processing_Im(single(u), net, 1, out_idx);
    z=im2double(z);

    err = norm(u-uprev,'fro')/norm(u,'fro');
    
    Res = [Res psnr(u,oimg)];
    imshow(u)
    drawnow
    erru=norm(u-oimg,'fro')/norm(oimg,'fro');
    
   
    if length(Res)>2 && Res(end)>Res(end-1)
        break;
    end
    if mod(nstep,20) == 0
        %disp(['Problem Step = ' num2str(nstep) ' ; Error = ' num2str(norm(u-oimg,'fro')/norm(oimg,'fro')) ' ; error = ' num2str(err)]);
    end
    
    if err <= tol
        %disp(['Problem Step = ' num2str(nstep) ' ; Error = ' num2str(norm(u-oimg,'fro')/norm(oimg,'fro')) ' ; error = ' num2str(err)]);
        break;
    end
end
%figure;plot(Res);title([num2str(modelSigma),' ',num2str(xi),' ',num2str(norm(u-oimg,'fro')/norm(oimg,'fro'))]);drawnow;
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