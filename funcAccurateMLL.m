function marginalacc = funcAccurateMLL(Npat, Nmix, Y, s0, sk, sx, priormu)

marginalacc = zeros(Npat,1);
for pat = 1:Npat
    nmix = Nmix(pat);
    X = cell(nmix,1);
    for i = 1:nmix
        X{i} = Y{pat, i};
    end
    As = zeros(nmix,1);
    Bs = zeros(nmix,1);
    logconsts = zeros(nmix,1);
    
    for j = 1:nmix
        ndata = length(X{j}); 
        sumx = sum(X{j});
        sumx2 = sum((X{j}).^2);
        multiplier = 0.5 / (ndata*sk + sx);
        logconsts(j) = 0.5*log(sx) - ndata * 0.5 * log(2*pi*sx) - 0.5 * log(ndata*sk + sx)...
            - 0.5 * sumx2 / sx + multiplier * sk/sx * sumx^2;
        As(j) = 0.5 / sk - sx/sk * multiplier;
        Bs(j) = multiplier * sumx;
    end
    A = sum(As);
    B = sum(Bs);
    C = sum(logconsts) + B^2 /A;
    mu = B/A;
    S = 0.5/A;
    marginalacc(pat) =  C + 0.5*log(S) - 0.5 * log(s0 + S)...
        - 0.5 * mu^2 / S - 0.5 * priormu^2 / s0 ...
        + 0.5 / (s0 + S) * (s0/S * mu^2 + S/s0 * priormu^2 + 2 * mu*priormu);
end
end
