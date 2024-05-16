function [u,Res] = EMRecon(P,Pt,f,c,N,oimg,opts)

if ~exist('opts','var'); opts = []; end

% Set up the option fields
[mit_out,tol,u_ini] = setopts(opts);

% normf = norm(f);

Const = Pt*ones(size(f));

% Initialization
if isempty(u_ini); u = ones(N,N); else u = u_ini; clear u_ini; end
Res = [];

%% Main Loop
for nstep = 1 : mit_out
    
    uprev = u;
    u = u.*reshape(Pt*(f./(P*u(:)+c))./Const,N,N);
    u = max(u,0);
    
    err = norm(u-uprev,'fro')/norm(u,'fro');
    Res = [Res norm(u-oimg,'fro')/norm(oimg,'fro')];
    
    if mod(nstep,20) == 0
%         disp(['Problem Step = ' num2str(nstep) '; Error = ' num2str(norm(u-oimg,'fro')/norm(oimg,'fro')) '; error = ' num2str(err)]);
    end
    
    if err <= tol
%         disp(['Problem Step = ' num2str(nstep) '; Error = ' num2str(norm(u-oimg,'fro')/norm(oimg,'fro')) '; error = ' num2str(err)]);
        break;
    end
    if length(Res)>2 && Res(end)>Res(end-1)
%         disp('bre')
        break;
        
    end
end

%% Subfunction
function [mit_out,tol,u_ini] = setopts(opts)

% define default option fields
mit_out = 200;
tol = 1.e-4;
u_ini = [];

% change to specified option fields if exist
if ~isempty(opts);
    if ~isa(opts,'struct'); error('EMRecon : opts not a struct'); end
    if isfield(opts,'mit_out'); mit_out = opts.mit_out; end
    if isfield(opts,'tol'); tol = opts.tol; end
    if isfield(opts,'u_ini'); u_ini = opts.u_ini; end
end
return;