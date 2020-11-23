function [lmlsplit, lmlmerge, lmlsplit_record, lmlmerge_record] = funcCalcSM1MarginalAccurate2(data_mll,allocation,counts,s0,sk,sx,globalmean,field_pca)

Npat = 2;
Nmix = counts;
Yall = cell(field_pca,1);
for j = 1:field_pca
    Yall{j,1} = cell(Npat, max(Nmix));
end
tempmixidx = ones(2,1);
for i = 1:length(allocation)
    idx1or2 = allocation(i);
    mixidx_in1or2 = tempmixidx(idx1or2);
    for j = 1:field_pca
        Yall{j,1}{idx1or2,mixidx_in1or2} = data_mll{i,1}(j,:);
    end
    tempmixidx(idx1or2) = mixidx_in1or2+1;
end

lmlsplit_record = zeros(field_pca,1);
for j = 1:field_pca
    lmlsplit_record(j) = funcAccurateMLL2(Npat, Nmix, Yall{j,1}, s0, sk, sx, globalmean(j));
end
lmlsplit = sum(lmlsplit_record);

Npat = 1;
Nmix = sum(counts);
Yall = cell(field_pca,1);
for j = 1:field_pca
    Yall{j,1} = cell(Npat, max(Nmix));
end
for i = 1:length(allocation)
    idx1or2 = 1;
    mixidx_in1or2 = i;
    for j = 1:field_pca
        Yall{j,1}{idx1or2,mixidx_in1or2} = data_mll{i,1}(j,:);
    end
end

lmlmerge_record = zeros(field_pca,1);
for j = 1:field_pca
    lmlmerge_record(j) = funcAccurateMLL2(Npat, Nmix, Yall{j,1}, s0, sk, sx, globalmean(j));
end
lmlmerge = sum(lmlmerge_record);

