function dir = dirichlet_sample(alpha)

dir = gamrnd(alpha, 1);
dir = dir ./ sum(dir);
