function H = haarmtx(n)
% k = 2^p + q
% p, specifies the magnitude and width (or scale) of the shape;
% q, specifies the position (or shift) of the shape.

H = zeros(n,n);
H(1,1:n) = ones(1,n)/sqrt(n);

for k = 1:n-1
    
    p = fix(log(k)/log(2));
    
    q = k-(2^p);
    
    k1 = 2^p;
    t1 = n/k1;
    
    k2 = 2^(p+1);
    t2 = n/k2;
    
    for i = 1:t2
        
        H(k+1,i+q*t1) = (2^(p/2))/sqrt(n);
        
        H(k+1,i+q*t1+t2) = -(2^(p/2))/sqrt(n);
        
    end
    
end