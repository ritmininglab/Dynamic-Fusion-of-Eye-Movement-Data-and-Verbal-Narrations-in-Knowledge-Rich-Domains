function [ Pi ] = SampleTransitionT0(transT0, H)





N = [transT0, 0];
Pi = dirichlet_sample(N + H);
