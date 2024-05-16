function Coeff = SWT(D,img,PatchSizeRow,PatchSizeCol)

[Row, Col] = size(img);

Coeff = zeros(Row-PatchSizeRow+1,Col-PatchSizeCol+1,size(D,2));

for j = 1 :size(D,2)
    kernel = reshape(D(:,j),[PatchSizeRow,PatchSizeCol]);
    Coeff(:,:,j) = filter2(kernel,img,'valid');
end