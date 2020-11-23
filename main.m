
clear all;
rng(1); 
tryBeam = 1;
ignoreTrans2phs = 0; 
updateEye = 1; 
updatetpcphs = 1;
updatebasiceye =1;
datatype = 4; 
maxIter = 5000;
phs3shrink = 0.5;
skipprintphscount = 20;

numimg = 30;

anchors = csvread('AnchorMatrix.csv');

useSM1 = 1;
useSM2 = 1;
printbasic = 0; 
print_basic2 = 0;
printSM = 0;
iterSM1skip = 1; 
iterSM2skip = 3; 

if datatype == 4 
    Eye = csvread('pcsMatrix.csv');
    Idx2PosMatrix = csvread('Idx2PosMatrix.csv');
    
    nonparametric = 1; 
    randominitial = 1; 
    markov = 1; 
    autoregress = 0;
    usesx = 0; 
    
    n_pt = 12;
    
    field_eye = 15;
    field_pcs = 15;
    startfield = 1;
    field_ar = field_pcs; 
    lag_eye = 0;
    
    sx = 0.6; 
    sk = 0.5; 
    s0 = 100; 
    
    df0 = field_ar + 25000; 
    S0 = sx *(df0-field_ar-1)*eye(field_ar);  
    kappa0 = sx/sk; 
    
    
    S00 = s0*eye(field_pcs);
    invS0 = inv(S0 / (df0-field_ar-1) ); 
    invS00 = inv(S00);
end




narrs = csvread('NarrationMatrixNoEye.csv');
n_phs = 3;
field_narr = 5;
n_chain = size(narrs,1)/field_narr;


alpha2 = 1/8; 
igamma2 = 1/16; 
blksample = 1;
csd_stickbreak2 = 1;


n_vocabulary = 789; 
n_tpc = 28;

MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs((i-1)*field_narr+1, 1:3);
end
Tmax = max(MB(:,3)); 

tpc_asgn = zeros(n_chain, Tmax);
for c=1:n_chain
    n_blk = MB(c,3);
    tpc_asgn(c,1:n_blk) = floor(1+(n_tpc-0.01)*(1:n_blk)/n_blk);
end
tpc_prob_vec = cell(n_chain,Tmax);

phs_asgn = zeros(n_chain, Tmax);
for c=1:n_chain
    n_blk = MB(c,3);
    phs_asgn(c,1:n_blk) = floor(1+(n_phs-phs3shrink)*(1:n_blk)/n_blk);
end
phs_prob_vec = cell(n_chain,Tmax);

total_obs = sum(MB(:,3));
array_narr = zeros(total_obs,1);
array_onehot = zeros(total_obs,n_vocabulary);
pos = 1;
for c=1:n_chain
    n_blk = MB(c,3);
    array_narr(pos:pos+n_blk-1,:) = (narrs((c-1)*field_narr+3, 4:3+n_blk))';
    pos = pos+n_blk;
end
for c=1:total_obs
    array_onehot(c,array_narr(c,1)) = 1;
end



a2 = 1/40;
b2 = 1; 
a3 = 5; 
b3 = 250; 


tau = a2*ones(n_tpc, n_vocabulary);
tau_prob = zeros(n_tpc, n_vocabulary);
for doc = 1:n_chain
    n_blk = MB(doc,3);
    for pos = 4:3+n_blk
        word = narrs(doc*field_narr-2,pos);
        tpc = tpc_asgn(doc,pos-3);
        tau(tpc,word) = tau(tpc,word)+1;
    end
end
for idx = 1:n_tpc
    tau_prob(idx,:) = dirichlet_sample(tau(idx,:));
end

transp = a3*ones(n_phs,n_phs)+b3*eye(n_phs);
transp_prob = zeros(n_phs,n_phs);
for doc = 1:n_chain
    n_blk = MB(doc,3);
    for blk = 1:n_blk-1
        p1 = phs_asgn(doc,blk);
        p2 = phs_asgn(doc,blk+1);
        transp(p1, p2) = transp(p1, p2)+1;
    end
end
for idx = 1:n_phs
    transp_prob(idx,:) = transp(idx,:) / sum(transp(idx,:));
end


if csd_stickbreak2==0
    pie = b2*ones(n_phs,n_tpc);
    pie_prob = zeros(n_phs,n_tpc);
    for doc=1:n_chain
        n_blk = MB(doc,3);
        for blk = 1:n_blk
            tpc = tpc_asgn(doc,blk);
            phs = phs_asgn(doc,blk);
            pie(phs,tpc) = pie(phs,tpc)+1;
        end
    end
    for idx = 1:n_phs
        pie_prob(idx,:) = pie(idx,:) / sum(pie(idx,:));
    end
    betavec2 = ones(1, n_tpc+1) / (n_tpc+1);
else
    pie = b2*ones(n_phs*numimg, n_tpc);
    for doc=1:n_chain
        n_blk = MB(doc,3);
        img = MB(doc,1);
        pos2 = (img-1)*n_phs; 
        for blk = 1:n_blk
            tpc = tpc_asgn(doc,blk);
            phs = phs_asgn(doc,blk);
            pie(pos2+phs,tpc) = pie(pos2+phs,tpc)+1;
        end
    end
    betavec2 = ones(1, n_tpc+1) / (n_tpc+1);
    betavec2 = topicHyperSample(pie, betavec2, alpha2, igamma2);
    pie_prob = SamplePhaseTopicMatrix(pie, alpha2 * betavec2);
    
end


mu0pcs = zeros(field_pcs,1); 


c0 = 0; 
d0 = 0; 
pi0 = 3.1415926;


alpha1 = 1/10; 
igamma1 = 1/20; 
igamma3 = 1/100; 

n_phs = 3;
n_chain = size(Eye,1)/field_eye;


nmap = sum(MB(:,3));


if autoregress ==1
    pos = 1;
    response = zeros(n_chain*Tmax, field_ar);
    for c=1:n_chain
        n_blk = Eye(c*field_eye,3);
        response(pos:pos+n_blk-lag_eye-1,:) = (Eye((c-1)*field_eye+startfield : (c-1)*field_eye+startfield+field_ar-1, 4+lag_eye : 3+n_blk))';
        pos = pos+n_blk-lag_eye;
    end
    response = response(1:pos-1,:);
    
    Idx = zeros(n_chain, Tmax-lag_eye);
    you = ones(n_chain*Tmax, lag_eye*field_ar);
    pos = 1;
    for c=1:n_chain
        n_blk = Eye(c*field_eye,3);
        for t=1:n_blk-lag_eye
            Idx(c,t) = pos;
            you(pos,1:field_ar*lag_eye) = reshape(Eye((c-1)*field_eye+startfield : (c-1)*field_eye+startfield+field_ar-1, 3+t : 3+t+lag_eye-1), 1, lag_eye*field_ar );
            pos = pos+1;
        end
    end
    you = you(1:pos-1,:);
    
else 
    
    pos = 1;
    response = zeros(nmap, field_pcs);
    Idx = zeros(n_chain, Tmax);
    for c=1:n_chain
        n_blk = MB(c,3);
        response(pos:pos+n_blk-lag_eye-1,:) = (Eye((c-1)*field_pcs+1:c*field_pcs, 4+lag_eye:3+n_blk))';
        Idx(c,1:n_blk) = pos:pos+n_blk-lag_eye-1;
        pos = pos+n_blk-lag_eye;
    end
    
    you = response;
end





pt_asgn = zeros(n_chain, Tmax);
for c=1:n_chain
    n_blk = MB(c,3);
    if randominitial==1
        pt_asgn(c,1:n_blk) = floor(1+rand(1,n_blk)*(n_pt-0.1));
    else
        pt_asgn(c,1:n_blk) = floor(1+(n_pt-0.1)*(1:n_blk)/n_blk);
    end
end
pt_prob_vec = cell(n_chain, Tmax);




transeye_cell = cell(n_phs,1);
transprobeye_cell = cell(n_phs,1);
transalt_cell = cell(n_phs, 1); 

trans_eye = c0*ones(n_pt,n_pt)+d0*eye(n_pt);
for j=1:n_phs
    transeye_cell{j,1} = trans_eye;
end
for doc = 1:n_chain
    n_blk = MB(doc,3);
    for blk = 1+lag_eye:n_blk-1
        phs = phs_asgn(doc,blk+1);
        p1 = pt_asgn(doc,blk);
        p2 = pt_asgn(doc,blk+1);
        transeye_cell{phs,1}(p1, p2) = transeye_cell{phs,1}(p1, p2)+1;
    end
end

trans_eye_t0 = c0*ones(1,n_pt);
for p1 = 1:n_pt
    trans_eye_t0(1, p1) = sum( pt_asgn(:,1+lag_eye) == p1 );
end
betavec1 = ones(1, n_pt+1) / (n_pt+1);
betavec1 = patternHyperSample([trans_eye_t0; funcConvert2Concat(transeye_cell,1)], betavec1, alpha1, igamma1);
for j=1:n_phs
    transprobeye_cell{j,1} = SampleTransitionMatrix(transeye_cell{j,1}, alpha1 * betavec1);
end
trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);


upperlimit = 32;

mx_meta = zeros(upperlimit,1);
mx_meta(1:n_pt,1) = 1*ones(n_pt,1);

mx_asgn = zeros(n_chain, Tmax);
for c=1:n_chain
    n_blk = MB(c,3);
    mx_asgn(c,1+lag_eye:n_blk) = ones(1, n_blk-lag_eye);
end

mx_count = zeros(upperlimit,upperlimit);
mx_beta = cell(upperlimit,1);


match1 = zeros(n_chain*160,1);
pos = 1;
for c=1:n_chain
    n_blk = MB(c,3);
    match1(pos:pos+n_blk-lag_eye-1,:) = (pt_asgn(c, lag_eye+1:n_blk))';
    pos = pos+n_blk-lag_eye;
end
match1 = match1(1:pos-1,:);
match2 = zeros(n_chain*160,1);
pos = 1;
for c=1:n_chain
    n_blk = MB(c,3);
    match2(pos:pos+n_blk-lag_eye-1,:) = (mx_asgn(c, lag_eye+1:n_blk))';
    pos = pos+n_blk-lag_eye;
end
match2 = match2(1:pos-1,:);


counts1=zeros(upperlimit,1);
if autoregress ==1
    
    mu0 = 1/lag_eye*repmat(eye(field_ar),1,lag_eye);
    covrowinv0 = S0/(0+df0-field_ar-1);
    covrow0 = inv(covrowinv0);
    covcol0 = matrix_normal_scale*eye(field_ar*lag_eye);
    covcolinv0 = inv(covcol0);
    
    mx_mu = cell(upperlimit,upperlimit);
    mx_covcol = cell(upperlimit,upperlimit);
    mx_covrow = cell(upperlimit,upperlimit);
    mx_covcolinv = cell(upperlimit,upperlimit);
    mx_covrowinv = cell(upperlimit,upperlimit);
    
    for p=1:n_pt
        for mx=1:mx_meta(p)+1 
            match3 = (match1==p) .*(match2==mx);
            response_p = (response(match3==1,:))';
            you_p = (you(match3==1,:))';
            mx_count(p,mx) = size(response_p,2); 
            if mx_count(p,mx)>0
                
                syy = response_p*response_p' + mu0 * covcolinv0 * mu0';
                syz = response_p*you_p' + mu0 * covcolinv0;
                szz = you_p*you_p' + covcolinv0;
                invszz = inv(szz);
                sss = syy - syz *invszz*syz';
                
                mx_mu{p,mx} = syz*invszz;
                mx_covrowinv{p,mx} = (sss+S0)/(mx_count(p,mx)+df0-field_ar-1);
                mx_covrow{p,mx} = inv(mx_covrowinv{p,mx});
                mx_covcolinv{p,mx} = szz;
                mx_covcol{p,mx} = invszz;
                
            else 
                
                mx_mu{p,mx} = mu0 + sqrt(covrow0)*randn(size(mu0))*sqrt(covcol0);
                mx_covrow{p,mx} = covrow0;
                mx_covrowinv{p,mx} = covrowinv0;
                mx_covcol{p,mx} = covcol0;
                mx_covcolinv{p,mx} = covcolinv0;
            end
        end
        mx_beta{p} = dirichlet_sample([mx_count(p,1:mx_meta(p)) igamma3]);
    end
    
else
    
    pat_param = cell(upperlimit,1); 
    pat_S = cell(upperlimit,1); 
    pat_sigma = cell(upperlimit,1); 
    pat_sigmainv = cell(upperlimit,1); 
    mx_param = cell(upperlimit,upperlimit); 
    mx_S = cell(upperlimit,upperlimit); 
    mx_sigmainv = cell(upperlimit,upperlimit); 
    
    for p=1:n_pt
        pat_sigmainv{p,1} = invS00;
        pat_sigma{p,1} = S00;
        pat_param{p,1} = mu0pcs;
    end
    
    for p=1:n_pt
        pat0pcs = pat_param{p,1};
        for mx=1:mx_meta(p)
            match3 = (match1==p) .*(match2==mx);
            response_p = response(match3==1,:);
            response_p = response_p';
            mx_count(p,mx) = size(response_p,2); 
            if mx_count(p,mx)>0
                n = mx_count(p,mx);
                xbar = mean(response_p,2);
                temp = response_p - repmat(xbar,1,mx_count(p,mx));
                scattermat = temp * temp';
                mx_param{p,mx} = kappa0/(kappa0+mx_count(p,mx)) * pat0pcs ...
                    + mx_count(p,mx)/(kappa0+mx_count(p,mx)) * xbar; 
                mx_S{p,mx} = S0 + scattermat + kappa0*n / (kappa0+n) * (xbar-pat0pcs)*(xbar-pat0pcs)';
                mx_sigmainv{p,mx} = inv(mx_S{p,mx} / (n+df0-field_ar-1));
            else
                mx_param{p,mx} = pat0pcs + sk*randn(field_pcs, 1);
                mx_S{p,mx} = S0;
                mx_sigmainv{p,mx} = invS0;
            end
        end
        mx_param{p,mx_meta(p)+1} = pat0pcs + sk*randn(field_pcs, 1);
        mx_S{p,mx_meta(p)+1} = S0;
        mx_sigmainv{p,mx_meta(p)+1} = invS0;
        
        mx_beta{p} = dirichlet_sample([mx_count(p,1:mx_meta(p)) igamma3]);
    end
    
    
    for p=1:n_pt
        temp1 = 0;
        temp2 = 0;
        temp3 = 0;
        for mx=1:mx_meta(p)
            temp2 = temp2 + mx_sigmainv{p,mx} / kappa0;
            temp3 = temp3 + mx_sigmainv{p,mx} / kappa0 * mx_param{p,mx};
        end
        pat_sigmainv{p,1} = inv(S00) + temp2;
        pat_sigma{p,1} = inv(pat_sigmainv{p,1});
        pat_param{p,1} = pat_sigma{p,1} * (temp3 + invS00 * mu0pcs);
    end
    
end




for iter = 1:maxIter
    
    if updatetpcphs ==1
        if csd_stickbreak2==1 
            
            u2 = ones(size(tpc_asgn));
            for doc=1:n_chain
                for t=1:MB(doc,3) 
                    
                    phs = phs_asgn(doc,t);
                    img = MB(doc,1);
                    pos2 = (img-1)*n_phs; 
                    u2(doc,t) = rand() * pie_prob(pos2+phs, tpc_asgn(doc,t));
                    
                end
            end
            min_u2 = min(min(u2));
            while max(pie_prob(:, end)) > min_u2     
                pl = size(pie_prob, 2);
                bl = size(betavec2, 2);
                assert(bl == pl);
                tau_prob(bl,:) = dirichlet_sample(ones(1,n_vocabulary)*a2);
                
                be = betavec2(end);
                bg = betarnd(1, igamma2);
                betavec2(bl) = bg * be;
                betavec2(bl+1) = (1-bg) * be;
                pe = pie_prob(:, end);
                
                a = repmat(alpha2 * betavec2(end-1), n_phs*numimg, 1);
                
                b = alpha2 * (1 - sum(betavec2(1:end-1)));
                pg = betarnd( a, b );
                if isnan(sum(pg))                   
                    pg = binornd(1, a./(a+b));
                end
                pie_prob(:, pl) = pg .* pe;
                pie_prob(:, pl+1) = (1-pg) .* pe;
            end
            if(size(pie_prob, 2)-1>n_tpc) && print_basic2==1
            end
            n_tpc = size(pie_prob, 2)-1;
        end
        
        
        
        for doc = 1:n_chain
            n_blk = MB(doc,3);
            for t = 1:n_blk 
                word = narrs(doc*field_narr-2,t+3);
                phs = phs_asgn(doc,t);
                
                img = MB(doc,1);
                pos2 = (img-1)*n_phs; 
                
                r = ((tau_prob(:,word))') .* pie_prob(pos2+phs, 1:n_tpc);
                r = r ./ sum(r);
                tpc_asgn(doc,t) = 1+sum(rand() > cumsum(r));
                tpc_prob_vec{doc,t} = r;
            end
        end
        
        
        if blksample==0
            for doc = 1:n_chain
                n_blk = MB(doc,3);
                for pos = 2:n_blk-1 
                    phs = phs_asgn(doc,pos);
                    tpc = tpc_asgn(doc,pos);
                    pre_phs = phs_asgn(doc,pos-1);
                    post_phs = phs_asgn(doc,pos+1);
                    
                    r = (pie_prob(:,tpc))'.*transp_prob(pre_phs,:).*((transp_prob(:,post_phs))');
                    r = r ./ sum(r);
                    phs_asgn(doc,pos) = 1+sum(rand() > cumsum(r));
                    phs_prob_vec{doc,pos} = r;
                end
            end
        else 
            if ignoreTrans2phs==1
                for doc = 1:n_chain
                    n_blk = MB(doc,3);
                    
                    backmsg = ones(n_phs, n_blk);
                    pos = n_blk-1;
                    backmsg(:,pos) = transp_prob(:,3);
                    backmsg(:,pos) = backmsg(:,pos)/sum(backmsg(:,pos));
                    
                    img = MB(doc,1);
                    pos2 = (img-1)*n_phs; 
                    temppie_prob = pie_prob(pos2+1:pos2+n_phs, :);
                    
                    for pos = n_blk-2:-1:2 
                        currentanchor = anchors(doc, pos);
                        if currentanchor ~=0 
                            continue;
                        end
                        nextanchor = anchors(doc, pos+1);
                        if nextanchor==0 
                            
                            nexttpc = tpc_asgn(doc,pos+1);
                            
                            backmsg(:,pos) = transp_prob*(temppie_prob(:,nexttpc).*backmsg(:,pos+1));
                            backmsg(:,pos) = backmsg(:,pos)/sum(backmsg(:,pos));
                        else
                            backmsg(:,pos) = transp_prob(:,nextanchor);
                            backmsg(:,pos) = backmsg(:,pos)/sum(backmsg(:,pos));
                        end
                    end
                    for pos = 2:n_blk-1
                        currentanchor = anchors(doc, pos);
                        if currentanchor ~=0 
                            phs_asgn(doc,pos) = currentanchor;
                        else
                            currenttpc = tpc_asgn(doc,pos);
                            pre_phs = phs_asgn(doc,pos-1);
                            
                            r = (transp_prob(pre_phs,:))' .* temppie_prob(:,currenttpc) .* backmsg(:,pos);
                            
                            r = r ./ sum(r);
                            phs_asgn(doc,pos) = 1+sum(rand() > cumsum(r));
                            phs_prob_vec{doc,pos} = r;
                        end
                    end
                end
            else
                for doc = 1:n_chain
                    n_blk = MB(doc,3);
                    
                    backmsg = ones(n_phs, n_blk);
                    pos = n_blk-1;
                    backmsg(:,pos) = transp_prob(:,3);
                    backmsg(:,pos) = backmsg(:,pos)/sum(backmsg(:,pos));
                    
                    img = MB(doc,1);
                    pos2 = (img-1)*n_phs; 
                    temppie_prob = pie_prob(pos2+1:pos2+n_phs, :);
                    
                    for pos = n_blk-2:-1:2
                        currentanchor = anchors(doc, pos);
                        if currentanchor ~=0 
                            continue;
                        end
                        nextanchor = anchors(doc, pos+1);
                        if nextanchor==0 
                            
                            nexttpc = tpc_asgn(doc,pos+1);
                            nextpt = pt_asgn(doc,pos+1);
                            pt = pt_asgn(doc,pos);
                            temp = zeros(n_phs,1);
                            for j = 1:n_phs
                                temp(j,1) = transprobeye_cell{j,1}(pt, nextpt);
                            end
                            
                            backmsg(:,pos) = transp_prob * ( temp.*  temppie_prob(:,nexttpc).*backmsg(:,pos+1));
                            backmsg(:,pos) = backmsg(:,pos)/sum(backmsg(:,pos));
                        else
                            backmsg(:,pos) = transp_prob(:,nextanchor);
                            backmsg(:,pos) = backmsg(:,pos)/sum(backmsg(:,pos));
                        end
                    end
                    for pos = 2:n_blk-1
                        currentanchor = anchors(doc, pos);
                        if currentanchor ~=0 
                            phs_asgn(doc,pos) = currentanchor;
                        else
                            currenttpc = tpc_asgn(doc,pos);
                            pre_phs = phs_asgn(doc,pos-1);
                            currentpt = pt_asgn(doc,pos);
                            pt = pt_asgn(doc,pos-1);
                            temp = zeros(n_phs,1);
                            for j = 1:n_phs
                                temp(j,1) = transprobeye_cell{j,1}(pt, currentpt);
                            end
                            
                            r = (transp_prob(pre_phs,:))' .* temp .* temppie_prob(:,currenttpc) .* backmsg(:,pos);
                            r = r ./ sum(r);
                            phs_asgn(doc,pos) = 1+sum(rand() > cumsum(r));
                            phs_prob_vec{doc,pos} = r;
                        end
                    end
                end
            end
            
        end
        
        if csd_stickbreak2==1
            zind2 = sort(setdiff(1:n_tpc, unique(tpc_asgn)));
            for i = size(zind2,2):-1:1
                if(size(zind2,2)>0) && print_basic2==1
                end
                betavec2(end) = betavec2(end) + betavec2(zind2(i));
                betavec2(zind2(i)) = [];
                pie_prob(:,zind2(i)) = [];
                mask = tpc_asgn > zind2(i);
                tpc_asgn(mask) = tpc_asgn(mask) - 1;
                n_tpc = n_tpc -1;
            end
            assert(size(pie_prob,2)-1 == n_tpc);
            assert(max(max(tpc_asgn)) == n_tpc);
        end
        
        
        tau = a2*ones(n_tpc, n_vocabulary);
        tau_prob = zeros(n_tpc, n_vocabulary);
        for doc = 1:n_chain
            n_blk = MB(doc,3);
            for pos = 4:3+n_blk
                word = narrs(doc*field_narr-2,pos);
                tpc = tpc_asgn(doc,pos-3);
                tau(tpc,word) = tau(tpc,word)+1;
            end
        end
        for idx = 1:n_tpc
            tau_prob(idx,:) = dirichlet_sample(tau(idx,:));
        end
        
        transp = a3*ones(n_phs,n_phs)+b3*eye(n_phs);
        transp_prob = zeros(n_phs,n_phs);
        for doc = 1:n_chain
            n_blk = MB(doc,3);
            for blk = 1:n_blk-1
                p1 = phs_asgn(doc,blk);
                p2 = phs_asgn(doc,blk+1);
                transp(p1, p2) = transp(p1, p2)+1;
            end
        end
        for idx = 1:n_phs
            transp_prob(idx,:) = transp(idx,:) / sum(transp(idx,:));
        end
        
        
        if csd_stickbreak2==0
            pie = b2*ones(n_phs,n_tpc);
            pie_prob = zeros(n_phs,n_tpc);
            for doc=1:n_chain
                n_blk = MB(doc,3);
                for blk = 1:n_blk
                    tpc = tpc_asgn(doc,blk);
                    phs = phs_asgn(doc,blk);
                    pie(phs,tpc) = pie(phs,tpc)+1;
                end
            end
            for idx = 1:n_phs
                pie_prob(idx,:) = pie(idx,:) / sum(pie(idx,:));
            end
            betavec2 = ones(1, n_tpc+1) / (n_tpc+1);
        else
            pie = b2*ones(n_phs*numimg, n_tpc);
            for doc=1:n_chain
                n_blk = MB(doc,3);
                img = MB(doc,1);
                pos2 = (img-1)*n_phs; 
                for blk = 1:n_blk
                    tpc = tpc_asgn(doc,blk);
                    phs = phs_asgn(doc,blk);
                    pie(pos2+phs,tpc) = pie(pos2+phs,tpc)+1;
                end
            end
            betavec2 = ones(1, n_tpc+1) / (n_tpc+1);
            betavec2 = topicHyperSample(pie, betavec2, alpha2, igamma2);
            pie_prob = SamplePhaseTopicMatrix(pie, alpha2 * betavec2);
            
        end
        
        
    end
    
    
    if updateEye>0
        if updatebasiceye ==1
            u1 = ones(size(pt_asgn,1),size(pt_asgn,2)+3); 
            for c=1:n_chain
                t=4+lag_eye; 
                if tryBeam==1
                    u1(c,t) = rand() * trans_prob_eye_t0(1, pt_asgn(c,t-3));
                else
                    u1(c,t) = rand();
                end
                for t=4+lag_eye+1:MB(c,3)+3 
                    phs = phs_asgn(c, t-3);
                    u1(c,t) = rand() * transprobeye_cell{phs,1}(pt_asgn(c,t-4), pt_asgn(c,t-3));
                end
            end
            min_u1 = min(min(u1));
            
            if nonparametric ==1
                if tryBeam==0
                    tempconcat = funcConvert2Concat(transprobeye_cell,1);
                else
                    [tempconcat, useless] = funcConvert2ConcatT0(transprobeye_cell, trans_prob_eye_t0,1);
                end
                while max(tempconcat(:, end)) > min_u1     
                    
                    pl = size(tempconcat, 2);
                    bl = size(betavec1,2);
                    assert(bl == pl);
                    
                    if tryBeam==0
                        added = zeros(n_phs*bl, bl);
                        for i=1:n_phs
                            added((i-1)*bl+1 : i*bl-1, :) = tempconcat((i-1)*(bl-1)+1:i*(bl-1), :);
                            added(i*bl,:) = dirichlet_sample(alpha1 * betavec1);
                        end
                        tempconcat = added;
                    else
                        
                        added = zeros(n_phs*bl +1, bl);
                        added(1,:) = tempconcat(1,:);
                        for i=1:n_phs
                            added((i-1)*bl+2 : i*bl, :) = tempconcat((i-1)*(bl-1)+2:i*(bl-1)+1, :);
                            added(i*bl+1,:) = dirichlet_sample(alpha1 * betavec1);
                        end
                        tempconcat = added;
                    end
                    
                    if autoregress==0
                        pat_sigmainv{pl,1} = invS00;
                        pat_sigma{pl,1} = S00;
                        pat_param{pl,1} = mu0pcs;
                        
                        mx_meta(pl) = 1;
                        mx_count(pl,1:2) = zeros(1,2);
                        mx_beta{pl,1} = dirichlet_sample([1,igamma3]);
                        
                        mx_param{pl,1} = pat_param{pl,1} + sk*randn(field_pcs, 1);
                        mx_S{pl,1} = S0;
                        mx_sigmainv{pl,1} = inv(mx_S{pl,1} / (0+df0-field_ar-1));
                    else
                        mx_meta(pl) = 1;
                        mx_count(pl,1:2) = zeros(1,2);
                        mx_beta{pl,1} = dirichlet_sample([1,igamma3]);
                        
                        mx_mu{pl,1} = mu0 + sqrt(covrow0)*randn(size(mu0))*sqrt(covcol0);
                        mx_covrow{pl,1} = covrow0;
                        mx_covrowinv{pl,1} = covrowinv0;
                        mx_covcol{pl,1} = covcol0;
                        mx_covcolinv{pl,1} = covcolinv0;
                    end
                    
                    
                    be = betavec1(end);
                    bg = betarnd(1, igamma1);
                    betavec1(bl) = bg * be;
                    betavec1(bl+1) = (1-bg) * be;
                    
                    pe = tempconcat(:, end);
                    
                    if tryBeam==0
                        a = repmat(alpha1 * betavec1(end-1), bl * n_phs, 1);
                    else
                        a = repmat(alpha1 * betavec1(end-1), bl * n_phs+1, 1);
                    end
                    b = alpha1 * (1 - sum(betavec1(1:end-1)));
                    pg = betarnd( a, b );
                    if isnan(sum(pg))                   
                        pg = binornd(1, a./(a+b));
                    end
                    tempconcat(:, pl) = pg .* pe;
                    tempconcat(:, pl+1) = (1-pg) .* pe;
                end
                if tryBeam==0
                    if(size(tempconcat,1)/n_phs > n_pt) && printbasic==1
                    end
                    transprobeye_cell = funcConvert2Concat(tempconcat,2);
                    n_pt = size(tempconcat, 1)/3;
                    assert(n_pt == size(betavec1,2) - 1);
                else
                    
                    if((size(tempconcat,1)-1)/n_phs > n_pt) && printbasic==1
                    end
                    [transprobeye_cell, trans_prob_eye_t0] = funcConvert2ConcatT0(tempconcat,0,2);
                    n_pt = (size(tempconcat,1)-1)/n_phs;
                    assert(n_pt == size(betavec1,2) - 1);
                    
                end
                
            end
            
            u3 = zeros(size(mx_asgn,1),size(mx_asgn,2)+3); 
            min_u3 = ones(n_pt,1);
            for c=1:n_chain
                for t=4+lag_eye:u3(c,3)+3 
                    p = pt_asgn(c,t-3);
                    mx = mx_asgn(c,t-3);
                    u3(c,t) = rand() * mx_beta{p,1}(1, mx);
                    if u3(c,t)<min_u3(p,1)
                        min_u3(p) = u3(c,t);
                    end
                end
            end
            
            if nonparametric ==1
                for p = 1:n_pt
                    while mx_beta{p,1}(1,end) > min_u3(p)     
                        
                        pl = size(mx_beta{p,1}, 2);
                        bl = pl;
                        
                        be = mx_beta{p,1}(end);
                        bg = betarnd(1, igamma3);
                        mx_beta{p,1}(bl) = bg * be;
                        mx_beta{p,1}(bl+1) = (1-bg) * be;
                        
                        
                        mx_param{p,bl+1} = pat_param{p,1} + sk*randn(field_pcs, 1);
                        mx_S{p,bl+1} = S0;
                        mx_sigmainv{p,bl+1} = invS0;
                    end
                    if(size(mx_beta{p,1}, 2)-1 > mx_meta(p)) && printbasic==1
                    end
                    mx_meta(p) = size(mx_beta{p,1}, 2)-1;
                end
                
            end
            
            
            
            
            
            for c = 1:n_chain
                image = MB(c,1);
                ppl = MB(c,2);
                n_blk = MB(c,3);
                dyn_prog = zeros(n_pt, n_blk+3);
                
                t = 1+3+lag_eye;
                if tryBeam==1
                    dyn_prog(:,4+lag_eye) = trans_prob_eye_t0(1,1:n_pt) > u1(c,4+lag_eye);
                else
                    dyn_prog(:,4+lag_eye) = ones(n_pt,1); 
                end
                idx_you = Idx(c,t-3-lag_eye);
                response_now = response(idx_you,:)';
                you_now = you(idx_you,1:field_ar*lag_eye)';
                pt_prob_vec{c,t} = ones(1,n_pt);
                
                
                for p=1:n_pt
                    temp = 0;
                    for mx = 1:mx_meta(p)
                        if autoregress==0
                            
                            temp = temp + 1000*mx_beta{p,1}(1,mx) * sqrt(det(mx_sigmainv{p,mx}))...
                                *exp(-0.5*(response_now - mx_param{p,mx})' * mx_sigmainv{p,mx} * (response_now - mx_param{p,mx}));
                        else
                            meannow = mx_mu{p,mx}*you_now;
                            if usesx==0
                                temp = temp + 1000 * mx_beta{p,1}(1,mx) / sqrt( det(mx_covrowinv{p,mx}) )...
                                    *exp(-0.5*(response_now - meannow)' * (mx_covrow{p,mx}) * (response_now - meannow));
                            else 
                                temp = temp + 1000 * mx_beta{p,1}(1,mx) * exp(-0.5 * (response_now - meannow)' *(eye(field_ar)/sx)* (response_now - meannow));
                            end
                        end
                    end
                    pt_prob_vec{c,t}(1,p) = temp;
                end
                
                pt_prob_vec{c,t} = pt_prob_vec{c,t}';
                pt_prob_vec{c,t} = pt_prob_vec{c,t}.*dyn_prog(:,t);
                dyn_prog(:,t) = pt_prob_vec{c,t} / sum(pt_prob_vec{c,t});
                
                
                for t = 4+lag_eye+1:n_blk+3
                    
                    idx_you = Idx(c,t-3-lag_eye);
                    response_now = response(idx_you,:)';
                    you_now = you(idx_you,1:field_ar*lag_eye)';
                    
                    phs = phs_asgn(c,t-3);
                    A = transprobeye_cell{phs,1}(1:n_pt, 1:n_pt) > u1(c,t);
                    dyn_prog(:,t) = A' * dyn_prog(:,t-1); 
                    
                    pt_prob_vec{c,t} = ones(1,n_pt);
                    
                    
                    for p=1:n_pt
                        temp = 0;
                        for mx = 1:mx_meta(p)
                            if autoregress==0
                                
                                temp = temp + 1000*mx_beta{p,1}(1,mx) * sqrt(det(mx_sigmainv{p,mx}))...
                                    *exp(-0.5*(response_now - mx_param{p,mx})' * mx_sigmainv{p,mx} * (response_now - mx_param{p,mx}));
                            else
                                meannow = mx_mu{p,mx}*you_now;
                                if usesx==0
                                    temp = temp + 1000 * mx_beta{p,1}(1,mx) / sqrt( det(mx_covrowinv{p,mx}) )...
                                        *exp(-0.5*(response_now - meannow)' * (mx_covrow{p,mx}) * (response_now - meannow));
                                else 
                                    temp = temp + 1000 * mx_beta{p,1}(1,mx) * exp(-0.5 * (response_now - meannow)' *(eye(field_ar)/sx)* (response_now - meannow));
                                end
                            end
                        end
                        pt_prob_vec{c,t}(1,p) = temp;
                    end
                    
                    if markov ==1
                        pt_prob_vec{c,t} = pt_prob_vec{c,t}';
                        pt_prob_vec{c,t} = pt_prob_vec{c,t}.*dyn_prog(:,t);
                        dyn_prog(:,t) = pt_prob_vec{c,t} / sum(pt_prob_vec{c,t});
                    else
                        dyn_prog(:,t) = (pt_prob_vec{c,t})' / sum(pt_prob_vec{c,t});
                    end
                    
                    r = dyn_prog(:,t);
                    if sum(r) ~= 0.0 && isfinite(sum(r))
                    else
                        return;
                    end
                end
                
                
                
                blk = n_blk+3;
                if markov==1
                    r = dyn_prog(:,blk);
                    if sum(r) ~= 0.0 && isfinite(sum(r))
                    else
                        return;
                    end
                    
                else
                    r = dyn_prog(:,blk);
                end
                
                
                r = r ./ sum(r);
                pt_asgn(c,blk-3) = 1+sum(rand() > cumsum(r));
                pt_prob_vec{c,blk} = r;
                p = pt_asgn(c,blk-3);
                idx_you = Idx(c,blk-3-lag_eye);
                response_now = response(idx_you,:)';
                
                
                you_now = you(idx_you,:)';
                
                mx_prob_vec = zeros(1,mx_meta(p));
                for mx=1:mx_meta(p)
                    if autoregress==0
                        
                        mx_prob_vec(1,mx) = 1000*mx_beta{p,1}(1,mx) * sqrt(det(mx_sigmainv{p,mx}))...
                            *exp(-0.5*(response_now - mx_param{p,mx})' * mx_sigmainv{p,mx} * (response_now - mx_param{p,mx}));
                    else
                        if usesx ==0
                            mx_prob_vec(1,mx) = 1000 * mx_beta{p,1}(1,mx) / sqrt(det(mx_covrowinv{p,mx}))...
                                *exp(-0.5*(response_now - meannow)' * (mx_covrow{p,mx}) * (response_now - meannow));
                        else
                            mx_prob_vec(1,mx) = 1000*mx_beta{p,1}(1,mx)*exp(-0.5*(response_now - meannow)' *(eye(field_ar)/sx)* (response_now - meannow));
                        end
                    end
                end
                mx_prob_vec = mx_prob_vec/sum(mx_prob_vec);
                mx_asgn(c,blk-3) = 1+sum(rand() > cumsum(mx_prob_vec));
                
                
                
                for blk = n_blk+3-1:-1:4+lag_eye
                    if markov==1
                        phsnext = phs_asgn(c, blk-3+1);
                        r = dyn_prog(:,blk) .* (transprobeye_cell{phsnext,1}(:, pt_asgn(c,blk+1-3)) > u1(c,blk+1));
                        if sum(r)==0
                            return;
                        end
                        
                    else
                        r = dyn_prog(:,blk);
                    end
                    r = r ./ sum(r);
                    pt_asgn(c,blk-3) = 1+sum(rand() > cumsum(r));
                    pt_prob_vec{c,blk} = r;            
                    p = pt_asgn(c,blk-3);
                    idx_you = Idx(c,blk-3-lag_eye);
                    response_now = response(idx_you,:)';
                    you_now = you(idx_you,:)';
                    
                    
                    mx_prob_vec = zeros(1,mx_meta(p));
                    for mx=1:mx_meta(p)
                        if autoregress==0
                            
                            mx_prob_vec(1,mx) = 1000 * mx_beta{p,1}(1,mx) * sqrt(det(mx_sigmainv{p,mx}))...
                                *exp(-0.5*(response_now - mx_param{p,mx})' * mx_sigmainv{p,mx} * (response_now - mx_param{p,mx}));
                        else
                            meannow = mx_mu{p,mx}*you_now;
                            if usesx ==0
                                mx_prob_vec(1,mx) = 1000 * mx_beta{p,1}(1,mx) / sqrt(det(mx_covrowinv{p,mx}))...
                                    *exp(-0.5*(response_now - meannow)' * (mx_covrow{p,mx}) * (response_now - meannow));
                            else
                                mx_prob_vec(1,mx) = 1000*mx_beta{p,1}(1,mx)*exp(-0.5*(response_now - meannow)' *(eye(field_ar)/sx)* (response_now - meannow));
                            end
                        end
                    end
                    mx_prob_vec = mx_prob_vec/sum(mx_prob_vec);
                    mx_asgn(c,blk-3) = 1+sum(rand() > cumsum(mx_prob_vec));
                end
            end
        end
        
        match1 = zeros(n_chain*160,1);
        pos = 1;
        for c=1:n_chain
            nblk = MB(c,3);
            match1(pos:pos-1+nblk-lag_eye,:) = (pt_asgn(c, lag_eye+1:nblk))';
            pos = pos+nblk-lag_eye;
        end
        match1 = match1(1:pos-1,:);
        match2 = zeros(n_chain*160,1);
        pos = 1;
        for c=1:n_chain
            nblk = MB(c,3);
            match2(pos:pos-1+nblk-lag_eye,:) = (mx_asgn(c, lag_eye+1:nblk))';
            pos = pos+nblk-lag_eye;
        end
        match2 = match2(1:pos-1,:);
        
        if autoregress==1
            
            for p=1:n_pt
                for mx=1:mx_meta(p)+1 
                    match3 = (match1==p) .*(match2==mx);
                    response_p = (response(match3==1,:))';
                    you_p = (you(match3==1,:))';
                    mx_count(p,mx) = size(response_p,2); 
                    if mx_count(p,mx)>0
                        
                        syy = response_p*response_p' + mu0 * covcolinv0 * mu0';
                        syz = response_p*you_p' + mu0 * covcolinv0;
                        szz = you_p*you_p' + covcolinv0;
                        invszz = inv(szz);
                        sss = syy - syz *invszz*syz';
                        
                        mx_mu{p,mx} = syz*invszz;
                        mx_covrowinv{p,mx} = (sss+S0)/(mx_count(p,mx)+df0-field_ar-1);
                        mx_covrow{p,mx} = inv(mx_covrowinv{p,mx});
                        mx_covcolinv{p,mx} = szz;
                        mx_covcol{p,mx} = invszz;
                    else
                        
                        mx_mu{p,mx} = mu0 + sqrt(covrow0)*randn(size(mu0))*sqrt(covcol0);
                        mx_covrow{p,mx} = covrow0;
                        mx_covrowinv{p,mx} = covrowinv0;
                        mx_covcol{p,mx} = covcol0;
                        mx_covcolinv{p,mx} = covcolinv0;
                    end
                end
                mx_beta{p} = dirichlet_sample([mx_count(p,1:mx_meta(p)) igamma3]);
            end
            
        else
            
            for p=1:n_pt
                pat0pcs = pat_param{p,1};
                for mx=1:mx_meta(p)
                    match3 = (match1==p) .*(match2==mx);
                    response_p = response(match3==1,:);
                    response_p = response_p';
                    mx_count(p,mx) = size(response_p,2); 
                    if mx_count(p,mx)>0
                        n = mx_count(p,mx);
                        xbar = mean(response_p,2);
                        temp = response_p - repmat(xbar,1,mx_count(p,mx));
                        scattermat = temp * temp';
                        mx_param{p,mx} = kappa0/(kappa0+mx_count(p,mx)) * pat0pcs ...
                            + mx_count(p,mx)/(kappa0+mx_count(p,mx)) * xbar; 
                        mx_S{p,mx} = S0 + scattermat + kappa0*n / (kappa0+n) * (xbar-pat0pcs)*(xbar-pat0pcs)';
                        mx_sigmainv{p,mx} = inv(mx_S{p,mx} / (n+df0-field_ar-1));
                    else
                        mx_param{p,mx} = pat0pcs + sk*randn(field_pcs, 1);
                        mx_S{p,mx} = S0;
                        mx_sigmainv{p,mx} = invS0;
                    end
                end
                mx_param{p,mx_meta(p)+1} = pat0pcs + sk*randn(field_pcs, 1);
                mx_S{p,mx_meta(p)+1} = S0;
                mx_sigmainv{p,mx_meta(p)+1} = invS0;
                
                mx_beta{p} = dirichlet_sample([mx_count(p,1:mx_meta(p)) igamma3]);
            end
            
            for p=1:n_pt
                temp1 = 0;
                temp2 = 0;
                temp3 = 0;
                for mx=1:mx_meta(p)
                    temp2 = temp2 + mx_sigmainv{p,mx} / kappa0;
                    temp3 = temp3 + mx_sigmainv{p,mx} / kappa0 * mx_param{p,mx};
                end
                pat_sigmainv{p,1} = inv(S00) + temp2;
                pat_sigma{p,1} = inv(pat_sigmainv{p,1});
                pat_param{p,1} = pat_sigma{p,1} * (temp3 + invS00 * mu0pcs);
            end
            
        end
        
        
        if nonparametric ==1
            zind1 = sort(setdiff(1:n_pt, unique(pt_asgn(:,1:end))));
            if(size(zind1,2)>0) && printbasic==1
            end
            
            for i = size(zind1,2):-1:1
                
                betavec1(end) = betavec1(end) + betavec1(zind1(i));
                betavec1(zind1(i)) = [];
                for phs = 1:n_phs
                    transprobeye_cell{phs,1}(:,zind1(i)) = [];
                    transprobeye_cell{phs,1}(zind1(i),:) = [];
                end
                if tryBeam==1
                    trans_prob_eye_t0(:,zind1(i)) = [];
                end
                temp = pt_asgn(:,1:end);
                temp(temp > zind1(i)) = temp(temp > zind1(i)) - 1;
                pt_asgn(:,1:end) = temp;
                
                mx_meta(zind1(i))=[];
                mx_count(zind1(i),:)=[];
                mx_beta(zind1(i),:)=[];
                
                pat_sigmainv(zind1(i),:)=[];
                pat_sigma(zind1(i),:)=[];
                pat_param(zind1(i),:)=[];
                
                if autoregress==0
                    mx_param(zind1(i),:)=[]; 
                    mx_S(zind1(i),:) = [];
                    mx_sigmainv(zind1(i),:) = [];
                else
                    mx_mu(zind1(i),:)=[];
                    mx_covrow(zind1(i),:)=[];
                    mx_covrowinv(zind1(i),:)=[];
                    mx_covcol(zind1(i),:)=[];
                    mx_covcolinv(zind1(i),:)=[];
                end
            end
            n_pt = size(transprobeye_cell{1,1},1);
            
            
            for p=1:n_pt
                zind1 = sort(setdiff(1:mx_meta(p), unique((pt_asgn==p).*mx_asgn)));
                if(size(zind1,2)>0) && printbasic==1
                end
                
                for i = size(zind1,2):-1:1
                    
                    mx_beta{p}(end) = mx_beta{p}(end) + mx_beta{p}(zind1(i));
                    mx_beta{p}(zind1(i)) = [];
                    temp = (pt_asgn==p).*mx_asgn(:,1:end);
                    mx_asgn(temp > zind1(i)) = mx_asgn(temp > zind1(i)) - 1;
                    
                    
                    
                    mx_count(p,zind1(i): mx_meta(p)-1) = mx_count(p, zind1(i)+1: mx_meta(p));
                    mx_count(p,mx_meta(p)) = 0;
                    mx_param(p,zind1(i): mx_meta(p)-1) = mx_param(p, zind1(i)+1: mx_meta(p));
                    mx_param{p,mx_meta(p)} = pat_param{p,1} + sk*randn(field_pcs,1);
                    mx_param{p,mx_meta(p)+1} = [];
                    mx_S(p,zind1(i): mx_meta(p)-1) = mx_S(p, zind1(i)+1: mx_meta(p));
                    mx_S{p,mx_meta(p)} = S0;
                    mx_S{p,mx_meta(p)+1} = [];
                    mx_sigmainv(p,zind1(i): mx_meta(p)-1) = mx_sigmainv(p, zind1(i)+1: mx_meta(p));
                    mx_sigmainv{p,mx_meta(p)} = invS0;
                    mx_sigmainv{p,mx_meta(p)+1} = [];
                    
                    mx_meta(p) = mx_meta(p)-1;
                    assert(mx_meta(p)==size(mx_beta{p},2)-1);
                    
                end
                if  min(mx_count(p,1:mx_meta(p))) ==0
                    return;
                end
            end
            
        end
        
        
        
        trans_eye = c0*ones(n_pt,n_pt)+d0*eye(n_pt);
        for j=1:n_phs
            transeye_cell{j,1} = trans_eye;
        end
        for doc = 1:n_chain
            n_blk = MB(doc,3);
            for blk = 1+lag_eye:n_blk-1
                phs = phs_asgn(doc,blk+1);
                p1 = pt_asgn(doc,blk);
                p2 = pt_asgn(doc,blk+1);
                transeye_cell{phs,1}(p1, p2) = transeye_cell{phs,1}(p1, p2)+1;
            end
        end

        trans_eye_t0 = c0*ones(1,n_pt);
        for p1 = 1:n_pt
            trans_eye_t0(1, p1) = sum( pt_asgn(:,1+lag_eye) == p1 );
        end
        betavec1 = ones(1, n_pt+1) / (n_pt+1);
        betavec1 = patternHyperSample([trans_eye_t0; funcConvert2Concat(transeye_cell,1)], betavec1, alpha1, igamma1);
        for j=1:n_phs
            transprobeye_cell{j,1} = SampleTransitionMatrix(transeye_cell{j,1}, alpha1 * betavec1);
        end
        trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);
        
        
        
        
        
        if useSM1==1 && mod(iter,iterSM1skip)==0
            [row,col] = find(pt_asgn>0);
            pick = ceil(rand()*length(row));
            r1 = row(pick);
            c1 = col(pick);
            pat1 = pt_asgn(r1,c1);
            mx1 = mx_asgn(r1,c1);
            
            locations = (pt_asgn>0) - (pt_asgn == pat1).*(mx_asgn == mx1);
            [row,col] = find(locations>0);
            if isempty(row)
                continue;
            end
            pick = ceil(rand()*length(row));
            r2 = row(pick);
            c2 = col(pick);
            pat2 = pt_asgn(r2,c2);
            mx2 = mx_asgn(r2,c2);
            
            isSplit1 = 0;
            if pat1 == pat2
                isSplit1 = 1;
            end
            if pat1 > pat2
                temp1 = [pat1, pat2];
                temp3 = [mx1, mx2];
                pat1 = temp1(2);
                pat2 = temp1(1);
                mx1 = temp3(2);
                mx2 = temp3(1);
            end
            
            
            if isSplit1 ==1
                
                patmus = zeros(field_pcs,3); 
                patvars = repmat(s0, field_pcs,3);
                totalmx = mx_meta(pat1);
                allocation = zeros(1,totalmx);
                allocation(mx1) = 1;
                allocation(mx2) = 2;
                
                
                data_mll = cell(totalmx,1);
                for mxidx = 1:totalmx
                    candi = (pt_asgn==pat1) .* (mx_asgn==mxidx);
                    [row,col] = find(candi==1);
                    idxs = zeros(size(row,1),1);
                    for i=1:size(row,1)
                        idxs(i) = Idx(row(i),col(i));
                    end
                    data_mll{mxidx,1} = (response(idxs, :) )';
                end
                
                qmerge = 0;
                qsplit = 0;
                temp = mean(data_mll{mx1,1},2);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 1);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 3);
                temp = mean(data_mll{mx2,1},2);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 2);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 3);
                
                counts = ones(1,2);
                for i = 1:length(allocation)
                    if i==mx1 || i==mx2
                        continue;
                    end
                    temp = mean(data_mll{i,1},2);
                    [ml1,ml2]= funcMC1proposeLL(patmus, patvars, temp, field_ar, sk);
                    qprob = counts(1)*ml1 / (counts(1)*ml1 + counts(2)*ml2);
                    if rand()<qprob
                        allocation(i)=1;
                        qsplit = qsplit + log(qprob);
                    else
                        allocation(i)=2;
                        qsplit = qsplit + log(1-qprob);
                    end
                    counts(allocation(i)) = counts(allocation(i))+1;
                    [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, allocation(i));
                    [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 3);
                end
                
                [lmlsplit, lmlmerge, lmlsplit_record, lmlmerge_record] = ...
                    funcCalcSM1MarginalAccurate2(data_mll,allocation,counts,s0,sk,sx,mu0pcs, field_pcs);
                
                npt = n_pt+1;
                pasgn = pt_asgn;
                for j = 1:length(allocation)
                    temp = zeros(size(pt_asgn));
                    if allocation(j)==2
                        temp = temp + (pt_asgn==pat1).*(mx_asgn==j);
                    end
                    pasgn(temp~=0) = npt;
                end
                
                transalt = c0*ones(npt,npt) + d0*eye(npt);
                for j=1:n_phs
                    transalt_cell{j,1} = transalt;
                end
                for doc = 1:n_chain
                    n_blk = MB(doc,3);
                    for blk = 1+lag_eye:n_blk-1
                        phs = phs_asgn(doc,blk+1);
                        p1 = pasgn(doc,blk);
                        p2 = pasgn(doc,blk+1);
                        transalt_cell{phs,1}(p1, p2) = transalt_cell{phs,1}(p1, p2)+1;
                    end
                end
                
                if tryBeam==0
                    betavec1alt = ones(1, npt+1) / (npt+1);
                    betavec1alt = patternHyperSample(funcConvert2Concat(transalt_cell,1), betavec1alt, alpha1, igamma1);
                    
                    pcsplit = funcPChigh2(transalt_cell, npt, alpha1, betavec1alt);
                    pcmerge = funcPChigh2(transeye_cell, n_pt, alpha1, betavec1);
                else
                    
                    transalt_t0 = 0*ones(1,npt);
                    for p1 = 1:npt
                        transalt_t0(1, p1) = sum( pasgn(:,1+lag_eye) == p1);
                    end
                    betavec1alt = ones(1, npt+1) / (npt+1);
                    [useful, useless] = funcConvert2ConcatT0(transalt_cell, transalt_t0,1);
                    betavec1alt = patternHyperSample(useful, betavec1alt, alpha1, igamma1);
                    
                    pcsplit = funcPChigh2t0(transalt_cell, transalt_t0, npt, alpha1, betavec1alt);
                    pcmerge = funcPChigh2t0(transeye_cell, trans_eye_t0, n_pt, alpha1, betavec1);
                end
                

                
                datacount = mx_count(pat1,1:mx_meta(pat1));
                [pcbiassplit, pcbiasmerge] = funcCalcPCbias(datacount, allocation, igamma3);
                
                
                logacceptance = (pcsplit - pcmerge + pcbiassplit - pcbiasmerge + lmlsplit - lmlmerge + qmerge - qsplit);
                acceptance = min(999, exp(min(10, logacceptance)));
                if rand()<=acceptance
                    if printSM==1
                    end
                    
                    pt_asgn = pasgn;
                    pat2 = n_pt + 1;
                    mxidx1=1;
                    mxidx2=1;
                    allocation2 = zeros(size(allocation));
                    for i=1:size(allocation,2)
                        if allocation(i)==1
                            allocation2(i)=mxidx1;
                            mxidx1 = mxidx1+1;
                        else
                            allocation2(i)=mxidx2;
                            mxidx2 = mxidx2+1;
                        end
                    end
                    oldmasgn = mx_asgn;
                    oldmparam = mx_param;
                    oldmS = mx_S;
                    oldmsigmainv = mx_sigmainv;
                    for i=1:size(allocation,2) 
                        if allocation(i) == 2
                            temp = (pt_asgn==pat2).*(oldmasgn==i);
                            mx_asgn(temp==1) = allocation2(i);
                            mx_param{pat2,allocation2(i)} = oldmparam{pat1,i};
                            mx_S{pat2,allocation2(i)} = oldmS{pat1,i};
                            mx_sigmainv{pat2,allocation2(i)} = oldmsigmainv{pat1,i};
                        end
                    end
                    
                    mx_param{pat2,mxidx2} = patmus(:,2);  
                    mx_S{pat2,mxidx2} = S0;
                    mx_sigmainv{pat2,mxidx2} = invS0;
                    
                    mx_meta(pat2) = mxidx2-1;
                    temp = mx_count(pat1,1:size(allocation,2));
                    temp(allocation==1) = [];
                    mx_count(pat2,1:size(temp,2)) = temp;
                    mx_count(pat2,size(temp,2)+1:end) = 0;
                    mx_beta{pat2} = dirichlet_sample([temp igamma3]);
                    
                    for p=pat2:pat2
                        temp2 = 0;
                        temp3 = 0;
                        for mx=1:mx_meta(p)
                            temp2 = temp2 + mx_sigmainv{p,mx} / kappa0;
                            temp3 = temp3 + mx_sigmainv{p,mx} / kappa0 * mx_param{p,mx};
                        end
                        pat_sigmainv{p,1} = inv(S00) + temp2;
                        pat_sigma{p,1} = inv(pat_sigmainv{p,1});
                        pat_param{p,1} = pat_sigma{p,1} * (temp3 + invS00 * mu0pcs);
                    end
                    
                    for i=1:size(allocation,2)
                        if allocation(i)==1
                            temp = (pt_asgn==pat1).*(oldmasgn==i);
                            mx_asgn(temp==1) = allocation2(i);
                            mx_param{pat1,allocation2(i)} = oldmparam{pat1,i};
                            mx_S{pat1,allocation2(i)} = oldmS{pat1,i};
                            mx_sigmainv{pat1,allocation2(i)} = oldmsigmainv{pat1,i};
                        end
                    end
                    mx_param{pat1,mxidx1} = patmus(:,1);  
                    mx_S{pat1,mxidx1} = S0;
                    mx_sigmainv{pat1,mxidx1} = invS0;
                    for i=mxidx1+1:size(mx_param,2)
                        mx_param{pat1,i} = [];
                        mx_S{pat1,i} = [];
                        mx_sigmainv{pat1,i} = [];
                    end
                    
                    mx_meta(pat1) = mxidx1-1;
                    temp = mx_count(pat1,1:size(allocation,2));
                    temp(allocation==2) = [];
                    mx_count(pat1,1:size(temp,2)) = temp;
                    mx_count(pat1,size(temp,2)+1:end) = 0;
                    mx_beta{pat1,1} = dirichlet_sample([temp igamma3]);
                    
                    for p=pat1:pat1
                        temp2 = 0;
                        temp3 = 0;
                        for mx=1:mx_meta(p)
                            temp2 = temp2 + mx_sigmainv{p,mx} / kappa0;
                            temp3 = temp3 + mx_sigmainv{p,mx} / kappa0 * mx_param{p,mx};
                        end
                        pat_sigmainv{p,1} = inv(S00) + temp2;
                        pat_sigma{p,1} = inv(pat_sigmainv{p,1});
                        pat_param{p,1} = pat_sigma{p,1} * (temp3 + invS00 * mu0pcs);
                    end
                    
                    n_pt = n_pt+1;

                    transeye_cell = transalt_cell;
                    betavec1 = betavec1alt;
                    for j=1:n_phs
                        transprobeye_cell{j,1} = SampleTransitionMatrix(transeye_cell{j,1}, alpha1 * betavec1);
                    end
                    if tryBeam==1
                        trans_eye_t0 = transalt_t0;
                        trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);
                    end
                end
            end
            
            
            
            
            
            if isSplit1 ==0 && n_pt >2
                
                patmus = zeros(field_pcs,3);
                patvars = repmat(s0, field_pcs,3);
                totalmx = mx_meta(pat1) + mx_meta(pat2);
                allocation = zeros(1,totalmx);
                allocation(1:mx_meta(pat1)) = 1;
                allocation(mx_meta(pat1)+1 : end) = 2;
                
                data_mll = cell(totalmx,1);
                for mxidx = 1:totalmx
                    if allocation(mxidx)==1
                        patnow = pat1;
                        mxnow = mxidx;
                    else
                        patnow = pat2;
                        mxnow = mxidx - mx_meta(pat1);
                    end
                    candi = (pt_asgn==patnow) .* (mx_asgn==mxnow);
                    [row,col] = find(candi==1);
                    idxs = zeros(size(row,1),1);
                    for i=1:size(row,1)
                        idxs(i) = Idx(row(i),col(i));
                    end
                    data_mll{mxidx,1} = (response(idxs, :) )';
                end
                
                qmerge = 0;
                qsplit = 0;
                temp = mean(data_mll{mx1,1},2);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 1);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 3);
                temp = mean(data_mll{mx_meta(pat1)+mx2,1},2);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 2);
                [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 3);
                
                
                counts = ones(1,2);
                for i = 1:length(allocation)
                    if i == mx1 || i == mx_meta(pat1)+mx2
                        continue;
                    end
                    if i > mx_meta(pat1)
                        patnow = pat2;
                        mxnow = i-mx_meta(pat1);
                    else
                        patnow = pat1;
                        mxnow = i;
                    end
                    temp = mean(data_mll{i,1},2);
                    [ml1,ml2]= funcMC1proposeLL(patmus, patvars, temp, field_ar, sk);
                    qprob = counts(1)*ml1 / (counts(1)*ml1 + counts(2)*ml2);
                    if allocation(i)==1
                        qsplit = qsplit + log(qprob);
                    else
                        qsplit = qsplit + log(1-qprob);
                    end
                    counts(allocation(i)) = counts(allocation(i))+1;
                    [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, allocation(i));
                    [patmus, patvars] = funcSeqUpdateSM1phi(patmus, patvars, sk, temp, 3);
                end
                
                [lmlsplit, lmlmerge, lmlsplit_record, lmlmerge_record] = ...
                    funcCalcSM1MarginalAccurate2(data_mll,allocation,counts,s0,sk,sx,mu0pcs,field_pcs);
                
                
                npt = n_pt-1;
                pasgn = pt_asgn;
                pasgn(pt_asgn==pat2) = pat1;
                pasgn(pt_asgn>pat2) = pasgn(pt_asgn>pat2)-1;
                
                
                transalt = c0*ones(npt,npt) + d0*eye(npt);
                for j=1:n_phs
                    transalt_cell{j,1} = transalt;
                end
                for doc = 1:n_chain
                    n_blk = MB(doc,3);
                    for blk = 1+lag_eye:n_blk-1
                        phs = phs_asgn(doc,blk+1);
                        p1 = pasgn(doc,blk);
                        p2 = pasgn(doc,blk+1);
                        transalt_cell{phs,1}(p1, p2) = transalt_cell{phs,1}(p1, p2)+1;
                    end
                end
                
                if tryBeam==0
                    
                    betavec1alt = ones(1, npt+1) / (npt+1);
                    betavec1alt = patternHyperSample(funcConvert2Concat(transalt_cell,1), betavec1alt, alpha1, igamma1);
                    
                    pcmerge = funcPChigh2(transalt_cell, npt, alpha1, betavec1alt);
                    pcsplit = funcPChigh2(transeye_cell, n_pt, alpha1, betavec1);
                else
                    
                    transalt_t0 = 0*ones(1,npt);
                    for p1 = 1:npt
                        transalt_t0(1, p1) = sum( pasgn(:,1+lag_eye) == p1);
                    end
                    betavec1alt = ones(1, npt+1) / (npt+1);
                    [useful, useless] = funcConvert2ConcatT0(transalt_cell, transalt_t0,1);
                    betavec1alt = patternHyperSample(useful, betavec1alt, alpha1, igamma1);
                    
                    pcmerge = funcPChigh2t0(transalt_cell, transalt_t0, npt, alpha1, betavec1alt);
                    pcsplit = funcPChigh2t0(transeye_cell, trans_eye_t0, n_pt, alpha1, betavec1);
                end
                
                
                
                datacount = [mx_count(pat1,1:mx_meta(pat1)), mx_count(pat2,1:mx_meta(pat2))];
                [pcbiassplit, pcbiasmerge] = funcCalcPCbias(datacount, allocation, igamma3);
                
                
                logacceptance = -(pcsplit - pcmerge + pcbiassplit - pcbiasmerge + lmlsplit - lmlmerge + qmerge - qsplit);
                acceptance = min(999, exp(min(10, logacceptance)));
                if rand()<acceptance
                    if printSM==1
                    end
                    n_pt = n_pt-1;
                    
                    bias = mx_meta(pat1);
                    bias2 = mx_meta(pat2);
                    mx_asgn(pt_asgn==pat2) = mx_asgn(pt_asgn==pat2)+bias;
                    mx_meta(pat1) = mx_meta(pat1) + mx_meta(pat2);
                    mx_count(pat1,bias+1:bias+bias2) = mx_count(pat2,1:bias2);
                    mx_param(pat1,bias+1:bias+bias2) = mx_param(pat2,1:bias2);
                    mx_param{pat1,bias+bias2+1} = patmus(:,3);
                    mx_S(pat1,bias+1:bias+bias2) = mx_S(pat2,1:bias2);
                    mx_sigmainv(pat1,bias+1:bias+bias2) = mx_sigmainv(pat2,1:bias2);
                    mx_S{pat1,bias+bias2+1} = S0;
                    mx_sigmainv{pat1,bias+bias2+1} = invS0;
                    
                    for p=pat1:pat1
                        temp2 = 0;
                        temp3 = 0;
                        for mx=1:mx_meta(p)
                            temp2 = temp2 + mx_sigmainv{p,mx} / kappa0;
                            temp3 = temp3 + mx_sigmainv{p,mx} / kappa0 * mx_param{p,mx};
                        end
                        pat_sigmainv{p,1} = inv(S00) + temp2;
                        pat_sigma{p,1} = inv(pat_sigmainv{p,1});
                        pat_param{p,1} = pat_sigma{p,1} * (temp3 + invS00 * mu0pcs);
                    end
                    
                    mx_beta{pat1} = dirichlet_sample([mx_count(pat1,1:bias+bias2) igamma3]);
                    mx_meta(pat2) = [];
                    mx_count(pat2:n_pt,:) = mx_count(pat2+1:n_pt+1,:);
                    mx_count(n_pt+1,:) = zeros(1,size(mx_count,2));
                    mx_beta(pat2,:) = [];
                    mx_param(pat2,:) = [];
                    mx_S(pat2,:) = [];
                    mx_sigmainv(pat2,:) = [];
                    pat_param(pat2,:) = [];
                    pat_sigma(pat2,:) = [];
                    pat_sigmainv(pat2,:) = [];
                    
                    pt_asgn = pasgn;
                    
                    transeye_cell = transalt_cell;
                    betavec1 = betavec1alt;
                    for j=1:n_phs
                        transprobeye_cell{j,1} = SampleTransitionMatrix(transeye_cell{j,1}, alpha1 * betavec1);
                    end
                    if tryBeam==1
                        trans_eye_t0 = transalt_t0;
                        trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);
                    end
                end
                
            end
        end
        
        
        
        
        if useSM2==1 && mod(iter, iterSM2skip)==0
            
            [row,col] = find(pt_asgn>0);
            pick = ceil(rand()*length(row));
            r1 = row(pick);
            c1 = col(pick);
            pat1 = pt_asgn(r1,c1);
            mx1 = mx_asgn(r1,c1);
            
            locations = pt_asgn == pat1;
            locations(r1,c1) = 0;
            [row,col] = find(locations>0);
            if isempty(row)
                continue;
            end
            pick = ceil(rand()*length(row));
            r2 = row(pick);
            c2 = col(pick);
            mx2 = mx_asgn(r2,c2);
            
            locations = (pt_asgn == pat1) .* ((mx_asgn==mx1) + (mx_asgn==mx2));
            locations(r1,c1) = 0;
            locations(r2,c2) = 0;
            [row,col] = find(locations>0);
            
            
            isSplit2 = 0;
            if mx1 == mx2
                isSplit2 = 1;
            end
            if mx1 > mx2
                temp1 = [mx1, mx2];
                temp2 = [r1,c1,r2,c2];
                mx1 = temp1(2);
                mx2 = temp1(1);
                r1 = temp2(3);
                c1 = temp2(4);
                r2 = temp2(1);
                c2 = temp2(2);
            end
            
            
            if isSplit2 ==1
                
                patmus = pat_param{pat1,1}; 
                mxmus = repmat(patmus, 1,3); 
                mxvars = repmat(sk, field_pcs,3);
                counts = ones(1,2); 
                allocation = zeros(size(row)); 
                
                qmerge = 0;
                qsplit = 0;
                lmlmerge = 0;
                lmlsplit = 0;
                data = (response(Idx(r1,c1), :) )';
                [ml1,~,ml3] = funcCalcMCgaussianLikelihood(mxmus,mxvars,data,field_ar,sx);
                lmlsplit = lmlsplit + log(ml1);
                lmlmerge = lmlmerge + log(ml3);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 1);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 3);
                data = (response(Idx(r2,c2), :) )';
                [ml1,ml2,ml3] = funcCalcMCgaussianLikelihood(mxmus,mxvars,data,field_ar,sx);
                lmlsplit = lmlsplit + log(ml2);
                lmlmerge = lmlmerge + log(ml3);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 2);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 3);
                
                for i=1:length(row)
                    data = (response(Idx(row(i), col(i)), :) )';
                    [ml1,ml2,ml3] = funcCalcMCgaussianLikelihood(mxmus,mxvars,data,field_ar,sx);
                    qprob = counts(1)*ml1 / (counts(1)*ml1 + counts(2)*ml2);
                    if rand()<qprob
                        allocation(i)=1;
                        qsplit = qsplit + log(qprob);
                        lmlsplit = lmlsplit + log(ml1);
                    else
                        allocation(i)=2;
                        qsplit = qsplit + log(1-qprob);
                        lmlsplit = lmlsplit + log(ml2);
                    end
                    counts(allocation(i)) = counts(allocation(i))+1;
                    lmlmerge = lmlmerge + log(ml3);
                    
                    [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, allocation(i));
                    [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 3);
                end
                
                pcmerge = logfactorial(sum(counts)-1);
                pcsplit = logfactorial(sum(counts(1))-1) + logfactorial(sum(counts(2))-1) + log(igamma3);
                
                
                logacceptance = (pcsplit - pcmerge + lmlsplit - lmlmerge + qmerge - qsplit);
                acceptance = min(999, exp(min(10, logacceptance)));
                if rand()<=acceptance
                    if printSM==1
                    end
                    
                    mx2 = mx_meta(pat1)+1;
                    mx_asgn(r2,c2) = mx2;
                    for i=1:length(row)
                        if allocation(i)==2
                            mx_asgn(row(i), col(i)) = mx2;
                        end
                    end
                    mx_param{pat1,mx1} = mxmus(:,1) + randn(field_pcs,1) .* sqrt(mxvars(1));
                    mx_param{pat1,mx2} = mxmus(:,2) + randn(field_pcs,1) .* sqrt(mxvars(2));
                    mx_param{pat1,mx2+1} = pat_param{pat1,1} + randn(field_pcs,1) * sqrt(sk);
                    
                    mx_S{pat1,mx1} = S0;
                    mx_S{pat1,mx2} = S0;
                    mx_S{pat1,mx2+1} = S0;
                    mx_sigmainv{pat1,mx1} = invS0;
                    mx_sigmainv{pat1,mx2} = invS0;
                    mx_sigmainv{pat1,mx2+1} = invS0;
                    
                    mx_meta(pat1) = mx2;
                    mx_count(pat1,mx1) = sum(allocation==1)+1;
                    mx_count(pat1,mx2) = sum(allocation==2)+1;
                    temp = mx_count(pat1,1:mx_meta(pat1));
                    mx_beta{pat1} = dirichlet_sample([temp igamma3]);
                end
            end
            
            
            
            if isSplit2 == 0
                
                patmus = pat_param{pat1,1}; 
                mxmus = repmat(patmus, 1,3); 
                mxvars = repmat(sk, field_pcs,3);
                counts = ones(1,2);
                allocation = zeros(size(row));
                
                for i=1:length(row)
                    if mx_asgn(row(i), col(i))==mx1
                        allocation(i) = 1;
                    elseif mx_asgn(row(i), col(i))==mx2
                        allocation(i) = 2;
                    else
                    end
                end
                
                qmerge = 0;
                qsplit = 0;
                lmlmerge = 0;
                lmlsplit = 0;
                data = (response(Idx(r1,c1), :) )';
                [ml1,~,ml3] = funcCalcMCgaussianLikelihood(mxmus,mxvars,data,field_ar,sx);
                lmlsplit = lmlsplit + log(ml1);
                lmlmerge = lmlmerge + log(ml3);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 1);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 3);
                data = (response(Idx(r2,c2), :) )';
                [ml1,ml2,ml3] = funcCalcMCgaussianLikelihood(mxmus,mxvars,data,field_ar,sx);
                lmlsplit = lmlsplit + log(ml2);
                lmlmerge = lmlmerge + log(ml3);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 2);
                [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 3);
                
                for i=1:length(row)
                    data = (response(Idx(row(i), col(i)), :) )';
                    [ml1,ml2,ml3] = funcCalcMCgaussianLikelihood(mxmus,mxvars,data,field_ar,sx);
                    
                    qprob = counts(1)*ml1 / (counts(1)*ml1 + counts(2)*ml2);
                    if allocation(i)==1
                        qsplit = qsplit + log(qprob);
                        lmlsplit = lmlsplit + log(ml1);
                    elseif allocation(i)==2
                        qsplit = qsplit + log(1-qprob);
                        lmlsplit = lmlsplit + log(ml2);
                    end
                    counts(allocation(i)) = counts(allocation(i))+1;
                    lmlmerge = lmlmerge + log(ml3);
                    
                    [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, allocation(i));
                    [mxmus, mxvars] = funcSeqUpdateSM2theta(mxmus, mxvars, sx, data, 3);
                end
                
                pcmerge = logfactorial(sum(counts)-1);
                pcsplit = logfactorial(sum(counts(1))-1)+logfactorial(sum(counts(2))-1) + log(igamma3);
                
                
                logacceptance = -(pcsplit - pcmerge + lmlsplit - lmlmerge + qmerge - qsplit);
                acceptance = min(999, exp(min(10, logacceptance)));
                if rand()<=acceptance
                    if printSM==1
                    end
                    
                    mx_asgn(r2,c2) = mx1;
                    for i=1:length(row)
                        if allocation(i)==2
                            mx_asgn(row(i), col(i)) = mx1;
                        end
                    end
                    temp = (pt_asgn==pat1) .*(mx_asgn>mx2);
                    mx_asgn(temp==1) = mx_asgn(temp==1) -1;
                    mx_param{pat1,mx1} = mxmus(:,3) + randn(field_pcs,1) .*sqrt(mxvars(3));
                    mx_param(pat1,mx2:mx_meta(pat1)) = mx_param(pat1,mx2+1:mx_meta(pat1)+1);
                    mx_param{pat1,mx_meta(pat1)+1} = [];
                    
                    mx_S{pat1,mx1} = S0;
                    mx_S(pat1,mx2:mx_meta(pat1)) = mx_param(pat1,mx2+1:mx_meta(pat1)+1);
                    mx_S{pat1,mx_meta(pat1)+1} = [];
                    mx_sigmainv{pat1,mx1} = invS0;
                    mx_sigmainv(pat1,mx2:mx_meta(pat1)) = mx_sigmainv(pat1,mx2+1:mx_meta(pat1)+1);
                    mx_sigmainv{pat1,mx_meta(pat1)+1} = [];
                    
                    mx_count(pat1,mx1) = length(allocation)+2;
                    mx_count(pat1,mx2:mx_meta(pat1)-1) = mx_count(pat1, mx2+1:mx_meta(pat1));
                    mx_count(pat1,mx_meta(pat1)) = 0;
                    mx_meta(pat1) = mx_meta(pat1)-1;
                    temp = mx_count(pat1,1:mx_meta(pat1));
                    mx_beta{pat1} = dirichlet_sample([temp igamma3]);
                    
                    
                    
                end
            end
            
            
        end
    end
    
end

pos = 1;
clusterresult = zeros(n_chain*160,1);
for c=1:n_chain
    n_blk = MB(c,3);
    clusterresult(pos:pos-1+n_blk-lag_eye,:) = (pt_asgn(c, lag_eye+1:n_blk))';
    pos = pos+n_blk-lag_eye;
end
clusterresult = clusterresult(1:pos-1);
csvwrite('ClusterResult.csv',clusterresult);

pos = 1;
clusterresult2 = zeros(n_chain*160,1);
for c=1:n_chain
    n_blk = MB(c,3);
    clusterresult2(pos:pos-1+n_blk-lag_eye,:) = (mx_asgn(c, lag_eye+1:n_blk))';
    pos = pos+n_blk-lag_eye;
end
clusterresult2 = clusterresult2(1:pos-1);
csvwrite('ClusterResultmx.csv',clusterresult2);

counts1 = zeros(n_pt,1);
for i =1:n_pt
    counts1(i,1) = sum(sum(pt_asgn(:,1+lag_eye:end)==i));
end
counts2 = zeros(n_tpc,1);
for i =1:n_tpc
    counts2(i,1) = sum(sum(tpc_asgn(:,1:end)==i));
end
counts3 = zeros(n_phs,1);
for i =1:n_phs
    counts3(i,1) = sum(sum(phs_asgn(:,1:end)==i));
end

topwords = zeros(n_tpc,n_vocabulary);
for i=1:n_tpc
    [value,idx] = sort(tau(i,:),'descend');
    topwords(i,:) = idx;
end
wordcounts = sum(tau)';
[topwordcounts, topwordidx] = sort(wordcounts,'descend');
csvwrite('popularwords.csv',[topwordidx,topwordcounts]);
csvwrite('topwords.csv',topwords);






match1 = zeros(n_chain*160,1);
pos = 1;
for c=1:n_chain
    n_blk = MB(c,3);
    match1(pos:pos-1+n_blk-lag_eye,:) = (pt_asgn(c, lag_eye+1:n_blk))';
    pos = pos+n_blk-lag_eye;
end
match1 = match1(1:pos-1,:);
match2 = zeros(n_chain*160,1);
pos = 1;
for c=1:n_chain
    n_blk = MB(c,3);
    match2(pos:pos-1+n_blk-lag_eye,:) = (mx_asgn(c, lag_eye+1:n_blk))';
    pos = pos+n_blk-lag_eye;
end
match2 = match2(1:pos-1,:);
for p=1:n_pt
    for mx=1:mx_meta(p)
        match3 = (match1==p) .*(match2==mx);
        response_p = response(match3==1,:);
        response_p = response_p';
        
        mx_count(p,mx) = size(response_p,2); 
    end
end
