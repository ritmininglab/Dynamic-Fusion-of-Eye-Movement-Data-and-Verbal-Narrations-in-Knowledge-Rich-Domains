function [mixmus, mixvars] = funcSeqUpdateSM2theta(mixmus, mixvars, sx, data, typeidx)



field_pca = size(mixmus,1);
for xory = 1:field_pca
oldvar = mixvars(xory, typeidx);
oldmu = mixmus(xory, typeidx);
mixvars(xory, typeidx) = 1/(1/sx + 1/oldvar);
mixmus(xory, typeidx) = mixvars(xory, typeidx) * (oldmu/oldvar + data(xory)/sx);
end

end
