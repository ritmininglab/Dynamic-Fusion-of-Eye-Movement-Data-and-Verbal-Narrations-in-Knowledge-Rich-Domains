function [ml1,ml2,ml3]= funcCalcMCgaussianLikelihood(mixmus,mixvars,data,field_ar,sx)



mls = ones(1,3);
field_pca = size(mixmus,1);
for typeidx=1:3
    for xory = 1:field_pca
        priormu = mixmus(xory,typeidx); 
        priorvar = mixvars(xory,typeidx); 
        datascalar = data(xory); 
        term3 = 1 / sqrt(priorvar + sx);
        term1 = - (datascalar^2)/(2*sx) - (priormu^2) / (2*priorvar);
        term2 = 0.5/(priorvar+sx) * ( (datascalar^2)*priorvar/sx ...
            + (priormu^2)*sx/priorvar + 2*datascalar*priormu);
        mls(typeidx) = mls(typeidx) * term3 * exp(term1+term2);
    end
end

ml1 = mls(1);
ml2 = mls(2);
ml3 = mls(3);

end
