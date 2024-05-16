function [gamma1,gamma2] = JointIsoThresh(alpha1,alpha2,beta)

Level = length(alpha1);
[nD,~] = size(alpha1{1});

for ki = 1 : Level
    vec = 0;
    for ji = 1 : nD
        for jj = 1 : nD
            if ((ji~=1 || jj~=1)) && sum(sum(beta{ki}{ji,jj}))~=0
                vec = vec + alpha1{ki}{ji,jj}.^2 + alpha2{ki}{ji,jj}.^2;
            end
        end
    end
    vec = sqrt(vec);
    for ji = 1 : nD
        for jj = 1 : nD
            if ((ji~=1) || (jj~=1)) && sum(sum(beta{ki}{ji,jj}))~=0
                gamma1{ki}{ji,jj}=max(vec-beta{ki}{ji,jj},0).*alpha1{ki}{ji,jj}./max(vec,eps);
                gamma2{ki}{ji,jj}=max(vec-beta{ki}{ji,jj},0).*alpha2{ki}{ji,jj}./max(vec,eps);
            else
                gamma1{ki}{ji,jj}=alpha1{ki}{ji,jj};
                gamma2{ki}{ji,jj}=alpha2{ki}{ji,jj};
            end
        end
    end
end