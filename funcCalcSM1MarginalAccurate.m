function [lmlsplit, lmlmerge, lmlsplit_record, lmlmerge_record] = funcCalcSM1MarginalAccurate(data_mll,allocation,counts,s0,sk,sx,globalmean)


Npat = 2;
Nmix = counts;
Yx = cell(Npat, max(Nmix));
Yy = cell(Npat, max(Nmix));
tempmixidx = ones(2,1);
for i = 1:length(allocation)
    idx1or2 = allocation(i);
    mix1or2 = tempmixidx(idx1or2); 
    Yx{idx1or2,mix1or2} = data_mll{i,1}(1,:);
    Yy{idx1or2,mix1or2} = data_mll{i,1}(2,:);
    tempmixidx(idx1or2) = mix1or2+1;
end
temp1 = funcAccurateMLL(Npat, Nmix, Yx, s0, sk, sx, globalmean(1));
temp2 = funcAccurateMLL(Npat, Nmix, Yy, s0, sk, sx, globalmean(2));

lmlsplit = sum(temp1) + sum(temp2);

Npat = 1;
Nmix = sum(counts);
Yx = cell(Npat, max(Nmix));
Yy = cell(Npat, max(Nmix));
for i = 1:length(allocation)
    idx1or2 = 1;
    mix1or2 = i;
    Yx{idx1or2,mix1or2} = data_mll{i,1}(1,:);
    Yy{idx1or2,mix1or2} = data_mll{i,1}(2,:);
end
temp3 = funcAccurateMLL(Npat, Nmix, Yx, s0, sk, sx, globalmean(1));
temp4 = funcAccurateMLL(Npat, Nmix, Yy, s0, sk, sx, globalmean(2));

lmlmerge = sum(temp3) + sum(temp4);

lmlsplit_record = [temp1;temp2];
lmlmerge_record = [temp3;temp4];
