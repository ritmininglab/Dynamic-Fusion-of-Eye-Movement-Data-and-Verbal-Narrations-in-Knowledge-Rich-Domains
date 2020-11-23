function [ml1,ml2]= funcMC1proposeLL(patmus, patvars, data, field_ar, sk)




mls = ones(1,2);
field_pca = size(patmus,1);
for typeidx = 1:2
    for xory = 1:field_pca
        priormu = patmus(xory,typeidx); 
        priorvar = patvars(xory,typeidx); 
        scalardata = data(xory, 1);
        term3 = 1 / sqrt( priorvar + sk );
        term1 = - (scalardata^2)/(2*sk) - (priormu^2) / (2*priorvar);
        term2 = 0.5/( priorvar+sk ) * ( (scalardata^2)*priorvar/sk ...
            + (priormu^2)*sk/priorvar + 2*scalardata*priormu );
        mls(typeidx) = mls(typeidx) * term3 * exp(term1+term2);
    end
end

ml1 = mls(1);
ml2 = mls(2);

end
