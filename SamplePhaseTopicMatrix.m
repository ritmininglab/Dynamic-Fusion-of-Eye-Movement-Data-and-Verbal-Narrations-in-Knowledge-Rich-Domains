function [ Pi ] = SamplePhaseTopicMatrix(counts, H)

N = zeros(size(counts,1),size(counts,2)+1);
N(:,1:end-1) = counts;
Pi = zeros(size(counts,1),size(counts,2)+1);

for k=1:size(counts,1)
    Pi(k, :) = dirichlet_sample(N(k,:) + H);
end
