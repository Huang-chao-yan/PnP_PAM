function [D,FilterSizeRow,FilterSizeCol] = InitializeFilter(FilterType,PatchSizeRow,PatchSizeCol)

switch FilterType
    
    case 'haar' % Haar framelet
        
        FilterSizeRow = PatchSizeRow;
        FilterSizeCol = PatchSizeCol;
        
        h1 = haarmtx(PatchSizeRow);
        h2 = haarmtx(PatchSizeCol);
        
        D = kron(h1,h2)';
    
    case 'ls'  % Piecewise linear
        
        FilterSizeRow = 3;
        FilterSizeCol = 3;
        
        h = [1/4 2/4 1/4 ;
            sqrt(2)/4 0 -sqrt(2)/4;
            -1/4 2/4 -1/4];
        
        D = kron(h,h)';
        
    case 'ls2'  % Two level piecewise linear
        
        FilterSizeRow = 7;
        FilterSizeCol = 7;
        
        h = [1/16 2/16 3/16 4/16 3/16 2/16 1/16 ;
            sqrt(2)/16 2*sqrt(2)/16 sqrt(2)/16 0 -sqrt(2)/16 -2*sqrt(2)/16 -sqrt(2)/16 ;
            -1/16 -2/16 1/16 4/16 1/16 -2/16 -1/16;
            sqrt(2)/4 0 -sqrt(2)/4 0 0 0 0 ;
            -1/4 2/4 -1/4 0 0 0 0 ;
            0 0 0 0 sqrt(2)/4 0 -sqrt(2)/4 ;
            0 0 0 0 -1/4 2/4 -1/4];
        
        D = kron(h,h)';
        
    case 'ls3'   % Three level piecewise linear
        
        FilterSizeRow = 15;
        FilterSizeCol = 15;
        
        h = [1/64 2/64 3/64 4/64 5/64 6/64 7/64 8/64 7/64 6/64 5/64 4/64 3/64 2/64 1/64 ;
            sqrt(2)/64 2*sqrt(2)/64 3*sqrt(2)/64 4*sqrt(2)/64 3*sqrt(2)/64 2*sqrt(2)/64 sqrt(2)/64 0 -sqrt(2)/64 -2*sqrt(2)/64 -3*sqrt(2)/64 -4*sqrt(2)/64 -3*sqrt(2)/64 -2*sqrt(2)/64 -sqrt(2)/64 ;
            -1/64 -2/64 -3/64 -4/64 -1/64 2/64 5/64 8/64 5/64 2/64 -1/64 -4/64 -3/64 -2/64 -1/64 ;
            sqrt(2)/16 2*sqrt(2)/16 sqrt(2)/16 0 -sqrt(2)/16 -2*sqrt(2)/16 -sqrt(2)/16 0 0 0 0 0 0 0 0 ;
            -1/16 -2/16 1/16 4/16 1/16 -2/16 -1/16 0 0 0 0 0 0 0 0 ;
            0 0 0 0 sqrt(2)/16 2*sqrt(2)/16 sqrt(2)/16 0 -sqrt(2)/16 -2*sqrt(2)/16 -sqrt(2)/16 0 0 0 0 ;
            0 0 0 0 -1/16 -2/16 1/16 4/16 1/16 -2/16 -1/16 0 0 0 0 ;
            0 0 0 0 0 0 0 0 sqrt(2)/16 2*sqrt(2)/16 sqrt(2)/16 0 -sqrt(2)/16 -2*sqrt(2)/16 -sqrt(2)/16 ;
            0 0 0 0 0 0 0 0 -1/16 -2/16 1/16 4/16 1/16 -2/16 -1/16 ;
            sqrt(2)/4 0 -sqrt(2)/4 0 0 0 0 0 0 0 0 0 0 0 0 ;
            -1/4 2/4 -1/4 0 0 0 0 0 0 0 0 0 0 0 0 ;
            0 0 0 0 0 0 sqrt(2)/4 0 -sqrt(2)/4 0 0 0 0 0 0 ;
            0 0 0 0 0 0 -1/4 2/4 -1/4 0 0 0 0 0 0 ;
            0 0 0 0 0 0 0 0 0 0 0 0 sqrt(2)/4 0 -sqrt(2)/4 ;
            0 0 0 0 0 0 0 0 0 0 0 0 -1/4 2/4 -1/4];
        
        D = kron(h,h)';
    
    case 'cs' % ONE level piecewise cubic, not good
        
        FilterSizeRow = 5;
        FilterSizeCol = 5;
        
        h = [1/16 4/16 6/16 4/16 1/16;
            1/8 2/8 0 -2/8 -1/8;
            -1/16*sqrt(6) 0 2/16*sqrt(6) 0 -1/16*sqrt(6);
            -1/8 2/8 0 -2/8 1/8;
            1/16 -4/16 6/16 -4/16 1/16];
        
        D = kron(h,h)';
        
    case 'cs2' % TWO level piecewise cubic
        
        FilterSizeRow = 13;
        FilterSizeCol = 13;
        
        h = [ 1/256 4/256 10/256 20/256 31/256 40/256 44/256 40/256 31/256 20/256 10/256 4/256 1/256 ;
            1/128 4/128 8/128 12/128 13/128 8/128 0 -8/128 -13/128 -12/128 -8/128 -4/128 -1/128 ;
            -sqrt(6)/256 -4*sqrt(6)/256 -6*sqrt(6)/256 -4*sqrt(6)/256 sqrt(6)/256 8*sqrt(6)/256 12*sqrt(6)/256 8*sqrt(6)/256 sqrt(6)/256 -4*sqrt(6)/256 -6*sqrt(6)/256 -4*sqrt(6)/256 -sqrt(6)/256 ;
            -1/128 -4/128 -4/128 4/128 11/128 8/128 0 -8/128 -11/128 -4/128 4/128 4/128 1/128 ;
            1/256 4/256 2/256 -12/256 -17/256 8/256 28/256 8/256 -17/256 -12/256 2/256 4/256 1/256 ;
            1/8 2/8 0 -2/8 -1/8 0 0 0 0 0 0 0 0 ;
            -1/16*sqrt(6) 0 2/16*sqrt(6) 0 -1/16*sqrt(6) 0 0 0 0 0 0 0 0 ;
            -1/8 2/8 0 -2/8 1/8 0 0 0 0 0 0 0 0 ;
            0 0 0 0 0 0 0 0 1/16 -4/16 6/16 -4/16 1/16 ;
            0 0 0 0 0 0 0 0 1/8 2/8 0 -2/8 -1/8 ;
            0 0 0 0 0 0 0 0 -1/16*sqrt(6) 0 2/16*sqrt(6) 0 -1/16*sqrt(6) ;
            0 0 0 0 0 0 0 0 -1/8 2/8 0 -2/8 1/8 ;
            0 0 0 0 0 0 0 0 1/16 -4/16 6/16 -4/16 1/16 ];
            
        D = kron(h,h)';
        
    case 'dct' % DCT (most general)
        
        FilterSizeRow = PatchSizeRow;
        FilterSizeCol = PatchSizeCol;
        
        h1 = dctmtx(PatchSizeRow);
        h2 = dctmtx(PatchSizeCol);
        
        D = kron(h1,h2)';
end