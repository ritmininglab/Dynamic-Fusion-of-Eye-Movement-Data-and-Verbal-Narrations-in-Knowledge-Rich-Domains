function [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, data, typeidx)



field_pca = size(patmus,1);
for xory = 1:field_pca
    oldvar = patvars(xory,typeidx); 
    oldmu = patmus(xory,typeidx); 
    posteriorvar = 1 / (1/sk + 1/oldvar); 
    posteriormu = posteriorvar * (oldmu/oldvar + data(xory)/sk); 
    patvars(xory, typeidx) = posteriorvar;
    patmus(xory, typeidx) = posteriormu;
end

end
