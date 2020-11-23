function [converted]= funcConvert2Concat(old, convert2concat)


if convert2concat ==1
converted = [old{1,1}; old{2,1}; old{3,1}];

else
converted = cell(3,1);
num_pattern = size(old,1)/3;
for i=1:3
converted{i,1} = old((i-1)*num_pattern+1 : i*num_pattern , :);
end
end


end
