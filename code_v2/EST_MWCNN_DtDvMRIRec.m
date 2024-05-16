function [u,v,D,Res] = EST_MWCNN_DtDvMRIRec(scaleNoi,xi,B,picks,lambda,mu,PatchSizeRow,PatchSizeCol,StepSizeRow,StepSizeCol,FilterType,uini,oimg,opts,alpha,beta,gamma,rho)

addpath(genpath('MWCNN\MWCNN-master/.'));

if ~exist('opts','var'); opts = []; end

[m,n] = size(B);
F = @(x) fft2(x);
FT = @(x) real(ifft2(x));
% F = @(x) fft2(x)/sqrt(m*n);
% FT = @(x) sqrt(m*n)*ifft2(x,'symmetric');
% Set up the option fields
[mit_learn,mit_inn,mit_out,tol] = setopts(opts);
% Pre-determine Low Pass Filter
p = PatchSizeRow*PatchSizeCol;
L = 1/sqrt(p)*ones(p,1);

% Initialization
u = uini;
z=uini;
[D,~,~] = InitializeFilter(FilterType,PatchSizeRow,PatchSizeCol);
Data = im2colstep(u,[PatchSizeRow,PatchSizeCol],[StepSizeRow,StepSizeCol]);
TmpData = (eye(p)-L*L')*Data;
Dtemp = D(:,2:end);
TmpCoeff = Dtemp'*Data;

for learnstep = 1 : mit_learn
    
    % Update Frames
    [X,~,Y] = svd(TmpData*TmpCoeff','econ');
    Dtemp = X*Y';

    % Update Coefficients
    TmpCoeff = wthresh(Dtemp'*Data,'h',lambda);

end
D = [L,Dtemp];
TmpCoeff = [L'*Data ; TmpCoeff];
v = SWT(D,u,PatchSizeRow,PatchSizeCol);
v(:,:,2:end) = wthresh(v(:,:,2:end),'h',lambda);
Res = [];
%disp('Initial Frames and Coefficients are Computed');

%% Main Loop
for nstep = 1 : mit_out
    
    uprev = u; TmpCoeffprev = TmpCoeff; vprev = v; Dprev = D;
    erprev=inf;
    % Solve u
    for innerstep = 1 : mit_inn
        
        uinnerprev = u;
        
        u = u-rho*(FT(picks.*(F(u)-B))+mu*(u-ISWT(D,v,m,n,PatchSizeRow,PatchSizeCol))+(xi+alpha)*(sqrt(2).*u-(z+uprev)./sqrt(2)));
        
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
    models=max(min(ceil(modelSigma/2)*2,50),2);
    load(fullfile('MWCNN\MWCNN-master\models\',num2str(models)));
    
    net = dagnn.DagNN.loadobj(net) ;
    net.removeLayer('objective') ;
    out_idx = net.getVarIndex('prediction') ;
    net.vars(net.getVarIndex('prediction')).precious = 1 ;
    net.mode = 'test';
    
    net.move('gpu');
    z=Processing_Im(single(u), net, 1, out_idx);
    z=im2double(z);
    
    Data = im2colstep(u,[PatchSizeRow,PatchSizeCol],[StepSizeRow,StepSizeCol]);
    TmpData = (eye(p)-L*L')*Data;
    CoeffTmp = TmpCoeff(2:end,:);
    
    [X,~,Y] = svd(TmpData*CoeffTmp'+beta/mu*Dprev(:,2:end),'econ');
    D = [L,X*Y'];
    
    TmpCoeff = (mu*D'*Data+gamma*TmpCoeffprev)/(mu+gamma);
    TmpCoeff(1,:) = L'*Data;
    TmpCoeff(2:end,:) = wthresh(TmpCoeff(2:end,:),'h',sqrt(2*lambda/(mu+gamma)));
    C = SWT(D,u,PatchSizeRow,PatchSizeCol);
    v = (mu*C+gamma*vprev)/(mu+gamma);
    v(:,:,1) = C(:,:,1);
    v(:,:,2:end) = wthresh(v(:,:,2:end),'h',sqrt(2*lambda/(mu+gamma)));
    
    err = norm(u-uprev,'fro')/norm(u,'fro');
    
    Res = [Res norm(u-oimg,'fro')/norm(oimg,'fro')];
    
    if mod(nstep,20) == 0
        %disp(['Problem Step = ' num2str(nstep) ' ; Error = ' num2str(norm(u-oimg,'fro')/norm(oimg,'fro')) ' ; error = ' num2str(err)]);
    end
    if length(Res)>2 && Res(end)>Res(end-1)
        break;
    end
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