function [ Pi ] = SampleTransitionMatrix(trans, H)

K = size(H,2);
Pi = zeros(K-1,K);

N = zeros(K);
N(1:K-1,1:K-1)= trans;

for k=1:K-1
    Pi(k, :) = dirichlet_sample(N(k,:) + H);
end
