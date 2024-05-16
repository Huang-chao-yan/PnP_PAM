function gamma=CoeffOper(op,alpha,beta,Gamma)
% This subroutine implement alpha op beta = gamma, where op= + - = * s n
Level=length(alpha);
[nD,nD1]=size(alpha{1});
for ki=1:Level
    if strcmp(op,'vs')
        vec=0;
        for ji=1:nD
            for jj=1:nD
                if (ji~=1 || jj~=1) && sum(sum(beta{ki}{ji,jj}))~=0
                    vec=vec+alpha{ki}{ji,jj}.^2;
                end
            end
        end
        vec=sqrt(vec);
    end
    for ji=1:nD
        for jj=1:nD
            if op=='='
                gamma{ki}{ji,jj}=alpha{ki}{ji,jj};
            elseif op=='-'
                if iscell(beta)
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj}-beta{ki}{ji,jj};
                else
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj}-beta;
                end
            elseif op=='+'
                if iscell(beta)
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj}+beta{ki}{ji,jj};
                else
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj}+beta;
                end
            elseif op=='*'
                if iscell(beta)
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj}.*beta{ki}{ji,jj};
                else
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj}.*beta;
                end
            elseif strcmp(op,'hm')
                if (ji~=1 || jj~=1)
                    if iscell(beta)
                        gamma{ki}{ji,jj}=alpha{ki}{ji,jj}.*beta{ki}{ji,jj};
                    else
                        gamma{ki}{ji,jj}=alpha{ki}{ji,jj}.*beta;
                    end
                else
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj};
                end
            elseif strcmp(op,'h')
                gamma{ki}{ji,jj}=wthresh(alpha{ki}{ji,jj},'h',beta{ki}{ji,jj});
            elseif strcmp(op,'s')
                gamma{ki}{ji,jj}=wthresh(alpha{ki}{ji,jj},'s',beta{ki}{ji,jj});
            elseif strcmp(op,'vs')
                if (ji~=1 || jj~=1) && sum(sum(beta{ki}{ji,jj}))~=0
                    gamma{ki}{ji,jj}=max(vec-beta{ki}{ji,jj},0).*alpha{ki}{ji,jj}./max(vec,eps);
                else
                    gamma{ki}{ji,jj}=alpha{ki}{ji,jj};
                end
            elseif strcmp(op,'cb')
                d=zeros(size(alpha{ki}{ji,jj}));
                d1=alpha{ki}{ji,jj};d2=beta{ki}{ji,jj};
                d(logical(Gamma))=d1(logical(Gamma));
                d(logical(1-Gamma))=d2(logical(1-Gamma));
                gamma{ki}{ji,jj}=d;
            end
        end
    end
end