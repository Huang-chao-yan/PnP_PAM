function Im_Out = ISWT(D,Coeff,Row,Col,FilterSizeRow,FilterSizeCol)

Im_Out = zeros([Row,Col]);
scalar = diag(D'*D);

for j = 1 : size(D,2)
    kernel = reshape(D(:,j),FilterSizeRow,FilterSizeCol);
    ker = kernel(:);
    ker = ker(FilterSizeRow*FilterSizeCol:-1:1);
    kernel = reshape(ker,FilterSizeRow,FilterSizeCol);
    Im_Out = Im_Out + filter2(kernel,Coeff(:,:,j)/scalar(j),'full');
end

for k = 1 : FilterSizeRow
    for k2 = 1 : FilterSizeCol
        if (k == 1)&(k2 == 1)
            mmask = ones(Row,Col);
        else
            if k == 1
                temp = zeros(Row,Col);
                temp(:,k2:Col-FilterSizeCol+k2-1) = 1;
                mmask = mmask + temp;
            elseif k2 == 1
                temp = zeros(Row,Col);
                temp(k:Row-FilterSizeRow+k-1,:) = 1;
                mmask = mmask + temp;
            else
                temp = zeros(Row,Col);
                temp(k:Row-FilterSizeRow+k-1,k2:Col-FilterSizeCol+k2-1) = 1;
                mmask = mmask + temp;
            end
        end
    end
end

mmask = double(mmask);
Im_Out = Im_Out./mmask;
% Im_Out = min(max(Im_Out,0),255);
% Im_Out = Im_Out/(FilterSizeRow*FilterSizeCol);