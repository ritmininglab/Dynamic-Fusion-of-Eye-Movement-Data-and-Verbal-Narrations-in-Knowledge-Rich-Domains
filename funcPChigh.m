function [logpc]= funcPChigh(transalt, npattern, alpha1, betavec1alt)

betau = 1-cumsum(betavec1alt);
temp = cumprod(betau); 
logpc = npattern * log(alpha1) + log(temp(npattern));

lognumerator = 0;
logdenominator = 0;
for i = 1:npattern 
    for j = 1:npattern 
        tempab = alpha1 * betavec1alt(j);
        for k = 1:transalt(i,j)
            lognumerator = lognumerator + log(tempab + k - 1);
        end
    end
end
for i = 1:npattern 
    for k = 1:sum(transalt(i,:)) 
        logdenominator = logdenominator + log(alpha1 + k - 1);
    end
end

logpc = logpc + lognumerator - logdenominator;
end
