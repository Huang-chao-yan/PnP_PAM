
function cn = cellnorm(x,type)

if (nargin == 1); type = 2; end

if iscell(x)
    Level    = length(x);
    [nD,~] = size(x{1});
    cn = 0;
    for ki=1:Level
        for ji=1:nD
            for jj=1:nD
                if ((ji~=1)||(jj~=1))||(ki==Level)
                    if (type==2)
                        cn = cn + norm(x{ki}{ji,jj},'fro')^2;
                    elseif (type==1)
                        cn = cn + sum(sum(abs(x{ki}{ji,jj})));
                    end
                end
            end
        end
    end
    if (type==2); cn = sqrt(cn); end
else
    if (type==2)
        cn = norm(x,'fro');
    elseif (type==1)
        cn = sum(sum(abs(x)));
    end
end
cn = full(cn);
