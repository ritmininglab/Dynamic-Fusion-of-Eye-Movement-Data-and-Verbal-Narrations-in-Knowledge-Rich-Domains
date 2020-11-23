function [Eye, MB_eye] = funcGenerateGMM(num_insobs,field_eye)

mubig = 4*[
    1 1;
    1 -1;
    -1 -1;
    -1 1;
    ]';
mudeviate = 1.5*[
    1 1;
    1 -1;
    -1 -1;
    -1 1;
    ]';
sigmaobservation = 1/2;

transprior = 1/16*ones(4,4) + 12/16*eye(4);

musmall = cell(4,4);
for bigc = 1:4
    for smallc = 1:4
        musmall{bigc,smallc} = mubig(:,bigc) + mudeviate(:,smallc);
    end
end

patternprior1 = zeros(num_insobs, 3+num_insobs);
patternprior1(:,3) = num_insobs;
patternprior2 = patternprior1;
for c=1:num_insobs
    patternprior1(c,4) = floor(1+rand()*3.99);
    for t=5:num_insobs+3
        oldidx = patternprior1(c,t-1);
        patternprior1(c,t) = 1 + sum(rand() > cumsum( transprior(oldidx,:) ));
    end
end

patternprior2(:,4:end) = floor(1+rand(num_insobs,num_insobs)*3.99);

Eye = zeros(num_insobs*field_eye, 3+num_insobs);
Eye(:,3) = num_insobs;
for c=1:num_insobs
    Eye((c-1)*field_eye+1:c*field_eye,1) = c;
    for t=4:num_insobs+3
        pat1 = patternprior1(c,t);
        pat2 = patternprior2(c,t);
        Eye(c*field_eye-5:c*field_eye-4,t) = musmall{pat1,pat2} + randn(2,1)*sigmaobservation;
    end
end

MB_eye = zeros(size(Eye,1)/field_eye,3);
for i =1:size(MB_eye,1)
    MB_eye(i,1:3) = Eye((i-1)*field_eye+1, 1:3);
end

end
