


rng(0);
maxIter = 100;
csd_basic = 1;
useSM1 = 1;
useSM2 = 1;
printbasic = 0;
printSM = 0;
iterSMskip = 1;
recordskip = 5;

asgn2param = 1; 

globalmean = [0;0];
sx = 0.5; 
sk = 1.5^2;
s0 = 4.5^2;

n__insobs = 50;
n__pt = 2;

field_eye = 7;
n__metadata = 3; 
field_ar = 2; 
lag_eye = 0;

[Eye, MB_eye] = funcGenerateGMM(n__insobs,field_eye);

n__chain = size(MB_eye,1);
Tmax_eye = max(Eye(:,3));

a0 = 0.1; 
b0 = 1; 
c0 = 0; 
d0 = 0; 
pi0 = 3.1415926;


alpha1 = 1/2; 
igamma1 = 1/8; 
igamma3 = 1/10; 

response = zeros(n__chain*160,field_ar);
pos = 1;
for c=1:n__chain
    nblk = Eye(c*field_eye,3);
    for i=4+lag_eye:3+nblk
        response(pos,:) = Eye((c-1)*field_eye+2:(c-1)*field_eye+3, i);
        pos = pos+1;
    end
end
response = response(1:pos-1,:);
Idx = zeros(n__chain, Tmax_eye-lag_eye);
you = ones(n__chain*160,lag_eye*field_ar);
pos = 1;
for c=1:n__chain
    nblk = Eye(c*field_eye,3);
    for t=1:nblk-lag_eye
        Idx(c,t) = pos;
        pos = pos+1;
    end
end
you = you(1:pos-1,:);





pt_asgn = zeros(n__chain, Tmax_eye);
for c=1:n__chain
    nblk = MB_eye(c,3);
    pt_asgn(c,1+lag_eye:nblk) = floor(1+(n__pt-0.1)*rand(1, nblk-lag_eye));
end
pt_prob_vec = cell(n__chain, Tmax_eye);

trans_eye = c0*ones(n__pt,n__pt)+d0*eye(n__pt);
for doc = 1:n__chain
    n__blk = MB_eye(doc,3);
    for blk = 4+lag_eye:3+n__blk-1
        p1 = pt_asgn(doc,blk-3);
        p2 = pt_asgn(doc,blk+1-3);
        trans_eye(p1, p2) = trans_eye(p1, p2)+1;
    end
end


trans_eye_t0 = c0*ones(1,n__pt);
for p1 = 1:n__pt
    trans_eye_t0(1, p1) = sum( pt_asgn(:,1+lag_eye) == p1);
end
betavec1 = ones(1, n__pt+1) / (n__pt+1);
betavec1 = patternHyperSample([trans_eye_t0; trans_eye], betavec1, alpha1, igamma1);
trans_prob_eye = SampleTransitionMatrix(trans_eye, alpha1 * betavec1);
trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);


match = zeros(n__chain*160,1);
pos = 1;
for c=1:n__chain
    nblk = Eye(c*field_eye,3);
    match(pos:pos-1+nblk-lag_eye,:) = (pt_asgn(c, lag_eye+1:nblk))';
    pos = pos+nblk-lag_eye;
end
match = match(1:pos-1,:);


upperlimit = n__insobs;

mx_meta = zeros(upperlimit,1);
mx_meta(1:n__pt,1) = 2*ones(n__pt,1);

mx_asgn = zeros(n__chain, Tmax_eye);
for c=1:n__chain
    n__blk = MB_eye(c,3);
    mx_asgn(c,1+lag_eye:n__blk) = floor(1+(rand(1,n__blk-lag_eye)>0.5));
end

mx_count = zeros(upperlimit,upperlimit);
pat_param = cell(upperlimit,1);
pat_var = cell(upperlimit,1);
mx_param = cell(upperlimit,upperlimit);
mx_var = cell(upperlimit,upperlimit);
mx_beta = cell(upperlimit,1);


match1 = zeros(n__chain*160,1);
pos = 1;
for c=1:n__chain
    nblk = MB_eye(c,3);
    match1(pos:pos-1+nblk-lag_eye,:) = (pt_asgn(c, lag_eye+1:nblk))';
    pos = pos+nblk-lag_eye;
end
match1 = match1(1:pos-1,:);
match2 = zeros(n__chain*160,1);
pos = 1;
for c=1:n__chain
    nblk = MB_eye(c,3);
    match2(pos:pos-1+nblk-lag_eye,:) = (mx_asgn(c, lag_eye+1:nblk))';
    pos = pos+nblk-lag_eye;
end
match2 = match2(1:pos-1,:);

for p=1:n__pt
    count = sum(match1==p);
    response_p = response(match1==p,:);
    response_p = response_p';
    if asgn2param ==1
        pat_param{p,1} = (globalmean+sum(response_p,2)) / (1+count); 
    else
        pat_param{p,1} = globalmean+sqrt(s0)*randn(2,1);
    end
    for mx=1:mx_meta(p)
        match3 = (match1==p) .*(match2==mx);
        response_p = response(match3==1,:);
        response_p = response_p';
        mx_count(p,mx) = size(response_p,2); 
        if asgn2param ==1
            if mx_count(p,mx)>0
                mx_param{p,mx} = (pat_param{p,1} + sum(response_p,2)) / (1+mx_count(p,mx)); 
            else
                mx_param{p,mx} = pat_param{p,1} + sqrt(sk)*randn(2,1);
            end
        else
            mx_param{p,mx} = pat_param{p,1} + sqrt(sk)*randn(2,1);
        end
    end
    mx_param{p,mx_meta(p)+1} = pat_param{p,1} + sqrt(sk)*randn(2,1);
    mx_beta{p} = dirichlet_sample([mx_count(p,1:mx_meta(p)) igamma3]);
end





recordmll = zeros(ceil(maxIter/recordskip),1);


for iter = 1:maxIter
    
    if csd_basic == 1
        u1 = ones(size(pt_asgn,1),size(pt_asgn,2)+3); 
        for c=1:n__chain
            t=4+lag_eye; 
            u1(c,t) = rand() * trans_prob_eye_t0(1, pt_asgn(c,t-3));
            for t=4+lag_eye+1:MB_eye(:,3)+3 
                u1(c,t) = rand() * trans_prob_eye(pt_asgn(c,t-4), pt_asgn(c,t-3));
            end
        end
        min_u1 = min(min(u1));
        
        tempconcat = [trans_prob_eye_t0; trans_prob_eye];
        while max(tempconcat(:,end)) > min_u1     
            
            pl = size(tempconcat, 2);
            bl = size(betavec1,2);
            assert(bl == pl);
            tempconcat(bl+1,:) = dirichlet_sample(alpha1 * betavec1);
            
            mx_meta(pl) = 1;
            mx_count(pl,1:2) = zeros(1,2);
            pat_param{pl,1} = globalmean+sqrt(s0)*randn(2,1);
            mx_param{pl,1} = pat_param{pl,1} + sqrt(sk)*randn(2,1);
            mx_param{pl,2} = pat_param{pl,1} + sqrt(sk)*randn(2,1);
            mx_beta{pl,1} = dirichlet_sample([1,igamma3]);
            
            be = betavec1(end);
            bg = betarnd(1, igamma1);
            betavec1(bl) = bg * be;
            betavec1(bl+1) = (1-bg) * be;
            
            pe = tempconcat(:, end);
            

            a = repmat(alpha1 * betavec1(end-1), bl+1, 1);
            b = alpha1 * (1 - sum(betavec1(1:end-1)));
            pg = betarnd( a, b );
            if isnan(sum(pg))                   
                pg = binornd(1, a./(a+b));
            end
            tempconcat(:, pl) = pg .* pe;
            tempconcat(:, pl+1) = (1-pg) .* pe;
        end
        
        trans_prob_eye = tempconcat(2:end,:);
        trans_prob_eye_t0 = tempconcat(1,:);
        
        if(size(trans_prob_eye,1) > n__pt) && printbasic==1
        end
        n__pt = size(trans_prob_eye, 1);
        assert(n__pt == size(betavec1,2) - 1);
        
        
        u3 = zeros(size(mx_asgn,1),size(mx_asgn,2)+3); 
        u3(:,1:3) = MB_eye(:,1:3);
        min_u3 = ones(n__pt,1);
        for c=1:n__chain
            for t=4+lag_eye:u3(c,3)+3 
                p = pt_asgn(c,t-3);
                mx = mx_asgn(c,t-3);
                u3(c,t) = rand() * mx_beta{p,1}(1, mx);
                if u3(c,t)<min_u3(p,1)
                    min_u3(p) = u3(c,t);
                end
            end
        end
        
        
        for p = 1:n__pt
            while mx_beta{p,1}(1,end) > min_u3(p)     
                
                pl = size(mx_beta{p,1}, 2);
                bl = pl;
                
                be = mx_beta{p,1}(end);
                bg = betarnd(1, igamma3);
                mx_beta{p,1}(bl) = bg * be;
                mx_beta{p,1}(bl+1) = (1-bg) * be;
                
                mx_param{p,bl+1} =  pat_param{p,1} + sqrt(sk)*randn(2,1);
            end
            if(size(mx_beta{p,1}, 2)-1 > mx_meta(p)) && printbasic==1
            end
            mx_meta(p) = size(mx_beta{p,1}, 2)-1;
        end
        
        
        
        for c = 1:n__chain
            image = MB_eye(c,1);
            ppl = MB_eye(c,2);
            n__blk = MB_eye(c,3);
            dyn_prog = zeros(n__pt, n__blk+3);
            
            t = 1+3+lag_eye;
            dyn_prog(:,4+lag_eye) = trans_prob_eye_t0(1,1:n__pt) > u1(c,4+lag_eye);
            
            idx_you = Idx(c,t-3-lag_eye);
            response_now = response(idx_you,:)';
            you_now = you(idx_you,1:field_ar*lag_eye)';
            pt_prob_vec{c,t} = ones(1,n__pt);
            for p=1:n__pt
                temp = 0;
                for mx = 1:mx_meta(p)+1
                    temp = temp+mx_beta{p,1}(1,mx)*exp(-0.5*(response_now - mx_param{p,mx})' * (eye(field_ar)/sx) * (response_now - mx_param{p,mx}));
                end
                pt_prob_vec{c,t}(1,p) = temp;
            end
            
            pt_prob_vec{c,t} = pt_prob_vec{c,t}';
            pt_prob_vec{c,t} = pt_prob_vec{c,t}.*dyn_prog(:,t);
            dyn_prog(:,t) = pt_prob_vec{c,t} / sum(pt_prob_vec{c,t});
            
            
            for t = 4+lag_eye+1:n__blk+3
                idx_you = Idx(c,t-3-lag_eye);
                response_now = response(idx_you,:)';
                you_now = you(idx_you,1:field_ar*lag_eye)';
                
                A = trans_prob_eye(1:n__pt, 1:n__pt) > u1(c,t);
                dyn_prog(:,t) = A' * dyn_prog(:,t-1); 
                
                pt_prob_vec{c,t} = ones(1,n__pt);
                
                for p=1:n__pt
                    temp = 0;
                    for mx = 1:mx_meta(p)+1
                        temp = temp+mx_beta{p,1}(1,mx)*exp(-0.5*(response_now - mx_param{p,mx})' * (eye(field_ar)/sx) * (response_now - mx_param{p,mx}));
                    end
                    pt_prob_vec{c,t}(1,p) = temp;
                end
                pt_prob_vec{c,t} = pt_prob_vec{c,t}';
                pt_prob_vec{c,t} = pt_prob_vec{c,t}.*dyn_prog(:,t);
                dyn_prog(:,t) = pt_prob_vec{c,t} / sum(pt_prob_vec{c,t});
                
                r = dyn_prog(:,t);
                if sum(r) ~= 0.0 && isfinite(sum(r))
                else
                    return;
                end
            end
            
            
            
            blk = n__blk+3;
            r = dyn_prog(:,blk);
            if sum(r) ~= 0.0 && isfinite(sum(r))
            else
                return;
            end
            r = r ./ sum(r);
            pt_asgn(c,blk-3) = 1+sum(rand() > cumsum(r));
            pt_prob_vec{c,blk} = r;
            p = pt_asgn(c,blk-3);
            idx_you = Idx(c,blk-3-lag_eye);
            response_now = response(idx_you,:)';
            
            mx_prob_vec = zeros(1,mx_meta(p));
            for mx=1:mx_meta(p)
                mx_prob_vec(1,mx) = mx_beta{p,1}(1,mx)* exp(-0.5*(response_now - mx_param{p,mx})' * (eye(field_ar)/sx) * (response_now - mx_param{p,mx}));
            end
            mx_prob_vec = mx_prob_vec/sum(mx_prob_vec);
            mx_asgn(c,blk-3) = 1+sum(rand() > cumsum(mx_prob_vec));
            
            
            for blk = n__blk+3-1:-1:4+lag_eye
                r = dyn_prog(:,blk) .* (trans_prob_eye(:, pt_asgn(c,blk+1-3)) > u1(c,blk+1));
                if sum(r)==0
                    return;
                end
                r = r ./ sum(r);
                pt_asgn(c,blk-3) = 1+sum(rand() > cumsum(r));
                pt_prob_vec{c,blk} = r;            
                p = pt_asgn(c,blk-3);
                idx_you = Idx(c,blk-3-lag_eye);
                response_now = response(idx_you,:)';
                
                mx_prob_vec = zeros(1,mx_meta(p));
                for mx=1:mx_meta(p)
                    mx_prob_vec(1,mx) = mx_beta{p,1}(1,mx)* exp(-0.5*(response_now - mx_param{p,mx})' * (eye(field_ar)/sx) * (response_now - mx_param{p,mx}));
                end
                mx_prob_vec = mx_prob_vec/sum(mx_prob_vec);
                mx_asgn(c,blk-3) = 1+sum(rand() > cumsum(mx_prob_vec));
            end
        end
        
        
        
        
        
        
        match1 = zeros(n__chain*160,1);
        pos = 1;
        for c=1:n__chain
            nblk = MB_eye(c,3);
            match1(pos:pos-1+nblk-lag_eye,:) = (pt_asgn(c, lag_eye+1:nblk))';
            pos = pos+nblk-lag_eye;
        end
        match1 = match1(1:pos-1,:);
        match2 = zeros(n__chain*160,1);
        pos = 1;
        for c=1:n__chain
            nblk = MB_eye(c,3);
            match2(pos:pos-1+nblk-lag_eye,:) = (mx_asgn(c, lag_eye+1:nblk))';
            pos = pos+nblk-lag_eye;
        end
        match2 = match2(1:pos-1,:);
        
        for p=1:n__pt
            count = sum(match1==p);
            response_p = response(match1==p,:);
            response_p = response_p';
            pat_param = funcResamplephi(p,s0,sk,globalmean, mx_meta,mx_count,mx_param,pat_param);
            
            for mx=1:mx_meta(p)
                match3 = (match1==p) .*(match2==mx);
                response_p = response(match3==1,:);
                response_p = response_p';
                
                mx_count(p,mx) = size(response_p,2); 
                mx_param = funcResampletheta(p,mx,sx,sk,response_p, mx_count,mx_param,pat_param);
            end
            mx_param{p,mx_meta(p)+1} = pat_param{p,1} + sqrt(sk)*randn(2,1);
            mx_beta{p} = dirichlet_sample([mx_count(p,1:mx_meta(p)) igamma3]);
            
            pat_param = funcResamplephi(p,s0,sk,globalmean, mx_meta,mx_count,mx_param,pat_param);
        end
        
        zind1 = sort(setdiff(1:n__pt, unique(pt_asgn(:,1:end))));
        if(size(zind1,2)>0) && printbasic==1
        end
        
        for i = size(zind1,2):-1:1
            betavec1(end) = betavec1(end) + betavec1(zind1(i));
            betavec1(zind1(i)) = [];
            trans_prob_eye(:,zind1(i)) = [];
            trans_prob_eye(zind1(i),:) = [];
            
            trans_prob_eye_t0(:,zind1(i)) = [];
            
            temp = pt_asgn(:,1:end);
            temp(temp > zind1(i)) = temp(temp > zind1(i)) - 1;
            pt_asgn(:,1:end) = temp;
            
            mx_meta(zind1(i))=[];
            mx_count(zind1(i),:)=[];
            mx_param(zind1(i),:)=[]; 
            mx_beta(zind1(i),:)=[];
            pat_param(zind1(i),:)=[];
        end
        n__pt = size(trans_prob_eye,1);
        
        for p=1:n__pt
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
                mx_param{p,mx_meta(p)} = pat_param{p,1} + sqrt(sk)*randn(2,1);
                mx_param{p,mx_meta(p)+1} = [];
                
                mx_meta(p) = mx_meta(p)-1;
            end
            assert(mx_meta(p)==size(mx_beta{p},2)-1);
            
            if  min(mx_count(p,1:mx_meta(p))) ==0
            end
        end
        
        
        trans_eye = c0*ones(n__pt,n__pt)+d0*eye(n__pt);
        for doc = 1:n__chain
            for blk = 4+lag_eye+1:MB_eye(doc,3)+3
                p1 = pt_asgn(doc,blk-1-3);
                p2 = pt_asgn(doc,blk-3);
                trans_eye(p1,p2) = trans_eye(p1,p2)+1;
            end
        end

        trans_eye_t0 = c0*ones(1,n__pt);
        for p1 = 1:n__pt
            trans_eye_t0(1, p1) = sum( pt_asgn(:,1+lag_eye) == p1);
        end
        betavec1 = patternHyperSample([trans_eye_t0; trans_eye], betavec1, alpha1, igamma1);
        trans_prob_eye = SampleTransitionMatrix(trans_eye, alpha1 * betavec1);
        trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);
        
    end
    
    
    if useSM1==1 && mod(iter,iterSMskip)==0
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
            
            patmus = zeros(2,3); 
            patvars = repmat(s0, 2,3);
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
            
            [lmlsplit, lmlmerge, lmlsplit_record, lmlmerge_record] = funcCalcSM1MarginalAccurate(data_mll,allocation,counts,s0,sk,sx,globalmean);
            
            npt = n__pt+1;
            pasgn = pt_asgn;
            for j = 1:length(allocation)
                temp = zeros(size(pt_asgn));
                if allocation(j)==2
                    temp = temp + (pt_asgn==pat1).*(mx_asgn==j);
                end
                pasgn(temp~=0) = npt;
            end
            transalt = 0*ones(npt, npt) + 0*eye(npt);
            for doc = 1:n__chain
                n__blk = MB_eye(doc,3);
                for blk = 1+lag_eye:n__blk-1
                    p1 = pasgn(doc,blk);
                    p2 = pasgn(doc,blk+1);
                    transalt(p1, p2) = transalt(p1, p2)+1;
                end
            end

            transalt_t0 = 0*ones(1,npt);
            for p1 = 1:npt
                transalt_t0(1, p1) = sum( pasgn(:,1+lag_eye) == p1);
            end
            betavec1alt = ones(1, npt+1) / (npt+1);
            betavec1alt = patternHyperSample([transalt_t0; transalt], betavec1alt, alpha1, igamma1);
            
            pcsplit = funcPChigh1t0([transalt_t0; transalt], npt, alpha1, betavec1alt);
            transold = ( trans_eye - d0*eye(n__pt) ) - c0;
            pcmerge = funcPChigh1t0([trans_eye_t0; transold], n__pt, alpha1, betavec1);
            
            datacount = mx_count(pat1,1:mx_meta(pat1));
            [pcbiassplit, pcbiasmerge] = funcCalcPCbias(datacount, allocation, igamma3);
            
            
            logacc = (pcsplit - pcmerge + pcbiassplit - pcbiasmerge + lmlsplit - lmlmerge + qmerge - qsplit);
            acc = min(999, exp(min(10, logacc)));
            if rand()<=acc
                if printSM==1
                end
                pt_asgn = pasgn;
                pat2 = n__pt + 1;
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
                for i=1:size(allocation,2) 
                    if allocation(i) == 2
                        temp = (pt_asgn==pat2).*(oldmasgn==i);
                        mx_asgn(temp==1) = allocation2(i);
                        mx_param{pat2,allocation2(i)} = oldmparam{pat1,i};
                        mx_var{pat2,allocation2(i)} = mx_var{pat1,i};
                    end
                end
                pat_param{pat2} = patmus(:,2);
                pat_var{pat2} = patvars(:,2);
                mx_param{pat2,mxidx2} = patmus(:,2);  
                mx_var{pat2,mxidx2} = patvars(:,2);
                
                mx_meta(pat2) = mxidx2-1;
                temp = mx_count(pat1,1:size(allocation,2));
                temp(allocation==1) = [];
                mx_count(pat2,1:size(temp,2)) = temp;
                mx_count(pat2,size(temp,2)+1:end) = 0;
                mx_beta{pat2} = dirichlet_sample([temp igamma3]);
                for i=1:size(allocation,2)
                    if allocation(i)==1
                        temp = (pt_asgn==pat1).*(oldmasgn==i);
                        mx_asgn(temp==1) = allocation2(i);
                        mx_param{pat1,allocation2(i)} = oldmparam{pat1,i};
                        mx_var{pat1,allocation2(i)} = mx_var{pat1,i};
                    end
                end
                pat_param{pat1} = patmus(:,1);
                pat_var{pat1} = patvars(:,1);
                mx_param{pat1,mxidx1} = patmus(:,1);
                for i=mxidx1+1:size(mx_param,2)
                    mx_param{pat1,i} = [];
                end
                mx_var{pat1,mxidx1} = patvars(:,1);
                
                mx_meta(pat1) = mxidx1-1;
                temp = mx_count(pat1,1:size(allocation,2));
                temp(allocation==2) = [];
                mx_count(pat1,1:size(temp,2)) = temp;
                mx_count(pat1,size(temp,2)+1:end) = 0;
                mx_beta{pat1,1} = dirichlet_sample([temp igamma3]);
                
                n__pt = n__pt+1;
                trans_eye = transalt;
                betavec1 = betavec1alt;
                trans_prob_eye = SampleTransitionMatrix(trans_eye, alpha1 * betavec1);
                
                trans_eye_t0 = transalt_t0;
                trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);
            end
        end
        
        
        
        
        
        if isSplit1 ==0 && n__pt>2
            
            patmus = zeros(2,3);
            patvars = repmat(s0, 2,3);
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
            
            [lmlsplit, lmlmerge, lmlsplit_record, lmlmerge_record] = funcCalcSM1MarginalAccurate(data_mll,allocation,counts,s0,sk,sx,globalmean);
            
            
            npt = n__pt-1;
            pasgn = pt_asgn;
            pasgn(pt_asgn==pat2) = pat1;
            pasgn(pt_asgn>pat2) = pasgn(pt_asgn>pat2)-1;
            transalt = 0*ones(npt, npt) + 0*eye(npt);
            for doc = 1:n__chain
                n__blk = MB_eye(doc,3);
                for blk = 1+lag_eye:n__blk-1
                    p1 = pasgn(doc,blk);
                    p2 = pasgn(doc,blk+1);
                    transalt(p1, p2) = transalt(p1, p2)+1;
                end
            end

            transalt_t0 = 0*ones(1,npt);
            for p1 = 1:npt
                transalt_t0(1, p1) = sum( pasgn(:,1+lag_eye) == p1);
            end
            betavec1alt = ones(1, npt+1) / (npt+1);
            betavec1alt = patternHyperSample([transalt_t0; transalt], betavec1alt, alpha1, igamma1);
            
            pcmerge = funcPChigh1t0([transalt_t0; transalt], npt, alpha1, betavec1alt);
            transold = ( trans_eye - d0*eye(n__pt) ) - c0;
            pcsplit = funcPChigh1t0([trans_eye_t0; transold], n__pt, alpha1, betavec1);
            
            
            datacount = [mx_count(pat1,1:mx_meta(pat1)), mx_count(pat2,1:mx_meta(pat2))];
            [pcbiassplit, pcbiasmerge] = funcCalcPCbias(datacount, allocation, igamma3);
            
            
            logacc = -(pcsplit - pcmerge + pcbiassplit - pcbiasmerge + lmlsplit - lmlmerge + qmerge - qsplit);
            acc = min(999, exp(min(10, logacc)));
            if rand()<acc
                if printSM==1
                end
                n__pt = n__pt-1;
                
                bias = mx_meta(pat1);
                bias2 = mx_meta(pat2);
                mx_asgn(pt_asgn==pat2) = mx_asgn(pt_asgn==pat2)+bias;
                mx_meta(pat1) = mx_meta(pat1) + mx_meta(pat2);
                mx_count(pat1,bias+1:bias+bias2) = mx_count(pat2,1:bias2);
                mx_param(pat1,bias+1:bias+bias2) = mx_param(pat2,1:bias2);
                mx_var(pat1,bias+1:bias+bias2) = mx_var(pat2,1:bias2);
                mx_param{pat1,bias+bias2+1} = patmus(:,3);
                mx_var{pat1,bias+bias2+1} = patvars(:,3);
                
                pat_param{pat1,1} = patmus(:,3);
                pat_var{pat1,1} = patvars(:,3);
                mx_beta{pat1} = dirichlet_sample([mx_count(pat1,1:bias+bias2) igamma3]);
                mx_meta(pat2) = [];
                mx_count(pat2:n__pt,:) = mx_count(pat2+1:n__pt+1,:);
                mx_count(n__pt+1,:) = zeros(1,size(mx_count,2));
                mx_beta(pat2,:) = [];
                mx_param(pat2,:) = [];
                mx_var(pat2,:) = [];
                pat_param(pat2,:) = [];
                pat_var(pat2,:) = [];
                
                pt_asgn = pasgn;
                
                trans_eye = transalt;
                betavec1 = betavec1alt;
                trans_prob_eye = SampleTransitionMatrix(trans_eye, alpha1 * betavec1);
                
                trans_eye_t0 = transalt_t0;
                trans_prob_eye_t0 = SampleTransitionT0(trans_eye_t0, alpha1 * betavec1);
            end
            
        end
    end
    
    
    
    
    
    if useSM2==1 && mod(iter, iterSMskip)==0
        
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
        
        
        
        if isSplit2 ==1 && mod(iter,iterSMskip)==0
            
            patmus = pat_param{pat1,1}; 
            mxmus = repmat(patmus, 1,3); 
            mxvars = repmat(sk, 2,3);
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
            
            
            logacc = (pcsplit - pcmerge + lmlsplit - lmlmerge + qmerge - qsplit);
            acc = min(999, exp(min(10, logacc)));
            if rand()<=acc
                if printSM==1
                end
                mx2 = mx_meta(pat1)+1;
                mx_asgn(r2,c2) = mx2;
                for i=1:length(row)
                    if allocation(i)==2
                        mx_asgn(row(i), col(i)) = mx2;
                    end
                end
                mx_param{pat1,mx1} = mxmus(:,1) + randn(2,1) .* sqrt(mxvars(1));
                mx_param{pat1,mx2} = mxmus(:,2) + randn(2,1) .* sqrt(mxvars(2));
                mx_param{pat1,mx2+1} = pat_param{pat1,1} + randn(2,1)*sqrt(sk);
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
            mxvars = repmat(sk, 2,3);
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
            
            
            logacc = -(pcsplit - pcmerge + lmlsplit - lmlmerge + qmerge - qsplit);
            acc = min(999, exp(min(10, logacc)));
            if rand()<=acc
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
                mx_param{pat1,mx1} = mxmus(:,3) + randn(2,1).*sqrt(mxvars(3));
                mx_param(pat1,mx2:mx_meta(pat1)) = mx_param(pat1,mx2+1:mx_meta(pat1)+1);
                mx_param{pat1,mx_meta(pat1)+1} = [];
                mx_count(pat1,mx1) = length(allocation)+2;
                mx_count(pat1,mx2:mx_meta(pat1)-1) = mx_count(pat1, mx2+1:mx_meta(pat1));
                mx_count(pat1,mx_meta(pat1)) = 0;
                mx_meta(pat1) = mx_meta(pat1)-1;
                temp = mx_count(pat1,1:mx_meta(pat1));
                mx_beta{pat1} = dirichlet_sample([temp igamma3]);
            end
        end
    end
    
    if mod(iter,recordskip) ==0
        data_mlls = cell(2,1);
        for j = 1:2
            data_mlls{j,1} = cell(n__pt,max(mx_meta));
        end
        for patnow = 1:n__pt
            for mxnow = 1:mx_meta(patnow)
                candi = (pt_asgn==patnow) .* (mx_asgn==mxnow);
                [row,col] = find(candi==1);
                idxs = zeros(size(row,1),1);
                for i=1:size(row,1)
                    idxs(i) = Idx(row(i),col(i));
                end
                for j = 1:2
                    data_mlls{j,1}{patnow, mxnow} = (response(idxs, j) )';
                end
            end
        end
        temp = 0;
        for j = 1:2
            temp = temp + sum(funcAccurateMLL(n__pt, mx_meta, data_mlls{j,1}, s0, sk, sx, globalmean(j)));
        end
        recordmll(iter/recordskip,1) = temp;
    end
    
end



save('GMMsm.mat');



counts1 = zeros(n__pt,1);
for i =1:n__pt
    counts1(i,1) = sum(sum(pt_asgn(:,1+lag_eye:end)==i));
end

match1 = zeros(n__chain*160,1);
pos = 1;
for c=1:n__chain
    nblk = MB_eye(c,3);
    match1(pos:pos-1+nblk-lag_eye,:) = (pt_asgn(c, lag_eye+1:nblk))';
    pos = pos+nblk-lag_eye;
end
match1 = match1(1:pos-1,:);
match2 = zeros(n__chain*160,1);
pos = 1;
for c=1:n__chain
    nblk = MB_eye(c,3);
    match2(pos:pos-1+nblk-lag_eye,:) = (mx_asgn(c, lag_eye+1:nblk))';
    pos = pos+nblk-lag_eye;
end
match2 = match2(1:pos-1,:);
for p=1:n__pt
    for mx=1:mx_meta(p)
        match3 = (match1==p) .*(match2==mx);
        response_p = response(match3==1,:);
        response_p = response_p';
        
        mx_count(p,mx) = size(response_p,2); 
    end
end

funcPlotResult(response, n__pt, match1, match2, mx_meta);

figure;
plot(1:length(recordmll),recordmll)
set(gcf, 'Position',  [100, 100, 400, 400]);
set(gca, 'Position',  [.1 .1 .85 .8]);
ylim([-10000 0])
yticks(-10000:2500:0)
xticks([])
t0 = title('Log Marginal','FontSize',15);
