function [converted, t0vecnew]= funcConvert2ConcatT0(old, t0vec, convert2concat)


if convert2concat ==1

converted = [t0vec; old{1,1}; old{2,1}; old{3,1}];
t0vecnew = 0; 

else


converted = cell(3,1);
num_pattern = (size(old,1)-1)/3; 
for i=1:3
converted{i,1} = old((i-1)*num_pattern+2 : i*num_pattern+1 , :);
end
end

t0vecnew = old(1, :);

end
