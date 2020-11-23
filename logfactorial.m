


function L2 = logfactorial(N,varargin)


    
    Nsort = N(:);   

    if any(fix(Nsort) ~= Nsort) || any(Nsort < 0) || ...
            ~isa(Nsort,'double') || ~isreal(Nsort)
        error('N must be a matrix of non-negative integers.')
    end
    
    if length(varargin) > 3
        error('Too many input arguments.')
    end
    
    
    defaultValues                 = {'gamma','
    nonemptyIdx                   = ~cellfun('isempty',varargin);
    defaultValues(nonemptyIdx)    = varargin(nonemptyIdx);
    [method fmt lengthOfSequence] = deal(defaultValues{:});
    
    
    switch lower(method)
        
        case 'gamma'	

            L = gammaln(N+1)/log(10);   
            
        case 'sum'      

            [Nsort,map] = sort(Nsort);
            Nsort = [0;Nsort];
            L = arrayfun(@sumlogIncrement, 1:numel(N));
            L = cumsum(L);  
            L(map) = L;     
            L = reshape(L,size(N));     

        otherwise

            error(['Method ''', method, ''' is unknown.'])

    end
	
	L2 = log(10) * L;
	
    if nargout > 1
        X = fix(L);      
        M = 10.^(L-X);   
    end
    
    if nargout > 3
        S = arrayfun(@makeString, 1:numel(N));
        S = reshape(S,size(N));     
    end
    function s = sumlogIncrement(j)
        lengthInterval = Nsort(j+1)-Nsort(j);
        a = Nsort(j);
        if lengthInterval <= lengthOfSequence
            s = sum(log10(a+(1:lengthInterval)));
        else
            seq = 1:lengthOfSequence;   
            rounds = floor(lengthInterval/lengthOfSequence);  
            partialSum = zeros(1,rounds);
            for r = 1:rounds
                partialSum(r) = sum(log10(seq+(r-1)*lengthOfSequence+a));
            end
            s = sum(partialSum) + ...
                sum(log10(((rounds*lengthOfSequence+1):lengthInterval)+a));
        end
    end

    function str = makeString(j)
        str = cellstr(strcat(num2str(M(j),fmt), 'e+', int2str(X(j))));
    end

end


