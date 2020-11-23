function sbeta = iHmmHyperSample(trans, ibeta, ialpha0, igamma)

K = length(ibeta)-1;        


N = trans;

M = zeros(K);
for j=1:K
    for k=1:K
        if N(j,k) == 0
            M(j,k) = 0;
        else
            for l=1:N(j,k)
                M(j,k) = M(j,k) + (rand() < (ialpha0 * ibeta(k)) / (ialpha0 * ibeta(k) + l - 1));
            end
        end
    end
end

ibeta = dirichlet_sample([sum(M,1) igamma]);

sbeta = ibeta;
