function [pcbiassplit, pcbiasmerge] = funcCalcPCbias(datacount, allocation, igamma3)


Nobs1 = sum(datacount(allocation==1));
Nobs2 = sum(datacount(allocation==2));
Nobsall = Nobs1 + Nobs2;
pcbiasmerge = 0;
pcbiassplit = 0;
for i = 1 : min(Nobs1,Nobs2)
    temp = log(igamma3+i-1);
    pcbiassplit = pcbiassplit - 2*temp;
    pcbiasmerge = pcbiasmerge - 1*temp;
end
for i = min(Nobs1,Nobs2)+1 : max(Nobs1,Nobs2)
    temp = log(igamma3+i-1);
    pcbiassplit = pcbiassplit - 1*temp;
    pcbiasmerge = pcbiasmerge - 1*temp;
end
for i = max(Nobs1,Nobs2)+1 : Nobsall
    temp = log(igamma3+i-1);
    pcbiasmerge = pcbiasmerge - 1*temp;
end

end
