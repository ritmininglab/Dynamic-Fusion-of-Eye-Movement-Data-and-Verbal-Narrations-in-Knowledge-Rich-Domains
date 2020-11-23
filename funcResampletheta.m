function mix_param = funcResampletheta(p,mix,sx,sk,response_p, mix_count,mix_param,pat_param)

snow = 1/(1/sk + mix_count(p,mix)/sx);
if mix_count(p,mix)>0
    mix_param{p,mix} = snow*(pat_param{p,1}/sk + sum(response_p,2)/sx);
else 
    mix_param{p,mix} = pat_param{p,1};
end

end
