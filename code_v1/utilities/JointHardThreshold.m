function [gamma1,gamma2] = JointHardThreshold(alpha1,alpha2,mu1,mu2,lambda)

Level = length(alpha1);
[nD,~] = size(alpha1{1});

for ki = 1 : Level
    for ji = 1 : nD
        for jj = 1 : nD
            if (ji~=1 || jj~=1)
                gamma1{ki}{ji,jj} = (mu1*alpha1{ki}{ji,jj}.^2 + mu2*alpha2{ki}{ji,jj}.^2>lambda{ki}{ji,jj}).*alpha1{ki}{ji,jj};
                gamma2{ki}{ji,jj} = (mu1*alpha1{ki}{ji,jj}.^2 + mu2*alpha2{ki}{ji,jj}.^2>lambda{ki}{ji,jj}).*alpha2{ki}{ji,jj};
            else
                gamma1{ki}{ji,jj} = alpha1{ki}{ji,jj};
                gamma2{ki}{ji,jj} = alpha2{ki}{ji,jj};
            end
        end
    end
end