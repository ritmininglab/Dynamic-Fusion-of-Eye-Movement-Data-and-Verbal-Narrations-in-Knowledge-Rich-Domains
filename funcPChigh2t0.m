function [logpc]= funcPChigh2t0(transalt_cell, transalt_t0, npattern, alpha1, betavec1alt)

c0 = 0;
d0 = 0;
for j = 1:3
transalt_cell{j,1} = transalt_cell{j,1} - c0*ones(npattern,npattern) - d0*eye(npattern);
end

tempconcat = funcConvert2ConcatT0(transalt_cell, transalt_t0,1);

betau = 1-cumsum(betavec1alt);
temp = cumprod(betau); 
logpc = npattern * log(alpha1) + log(temp(npattern-1));

lognumerator = 0;
logdenominator = 0;
for i = 1 : 3*npattern+1 
    for j = 1:npattern 
        tempab = alpha1 * betavec1alt(j);
        for k = 1:tempconcat(i,j)
            lognumerator = lognumerator + log(tempab + k - 1);
        end
    end
end
for i = 1 : 3*npattern+1 
    for k = 1:sum(tempconcat(i,:)) 
        logdenominator = logdenominator + log(alpha1 + k - 1);
    end
end

logpc = logpc + lognumerator - logdenominator;
end
