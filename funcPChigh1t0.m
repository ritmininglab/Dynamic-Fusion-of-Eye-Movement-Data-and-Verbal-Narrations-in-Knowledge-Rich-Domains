function [logpc]= funcPChigh1t0(transalt, npattern, alpha1, betavec1alt)




betau = 1-cumsum(betavec1alt);
temp = cumprod(betau);
logpc = npattern * log(alpha1) + log(temp(npattern-1));

lognumerator = 0;
logdenominator = 0;
for i = 1:npattern+1 
    for j = 1:npattern 
        tempab = alpha1 * betavec1alt(j);
        for k = 1:transalt(i,j)
            lognumerator = lognumerator + log(tempab + k - 1);
        end
    end
end
for i = 1:npattern+1 
    for k = 1:sum(transalt(i,:)) 
        logdenominator = logdenominator + log(alpha1 + k - 1);
    end
end

logpc = logpc + lognumerator - logdenominator;
end
