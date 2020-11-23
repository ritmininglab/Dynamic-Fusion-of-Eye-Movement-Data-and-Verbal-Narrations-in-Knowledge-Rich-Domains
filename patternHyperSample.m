function sbeta = patternHyperSample(N, ibeta, ialpha0, igamma)

K = length(ibeta)-1;        



M = zeros(size(N));
for j=1:size(N,1) 
    for k=1:size(N,2) 
		if N(j,k) == 0
            M(j,k) = 0;
        for l=1:N(j,k)
            M(j,k) = M(j,k) + (rand() < (ialpha0 * ibeta(k)) / (ialpha0 * ibeta(k) + l - 1));
        end
    end
end

sbeta = dirichlet_sample([sum(M,1) igamma]);

