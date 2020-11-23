function pat_param = funcResamplephi(p,s0,sk,globalmean, mix_meta,mix_count,mix_param,pat_param)

snow = 1/(1/s0 + mix_meta(p)/sk);
temp = 0;
for mix=1:mix_meta(p)
	temp = temp + mix_param{p,mix};
end
pat_param{p,1} = snow*(globalmean/s0 + temp/sk); 
end
