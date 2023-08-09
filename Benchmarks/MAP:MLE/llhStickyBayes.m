function llh = llhStickyBayes(p,beh, prior)

beta = 10*p(1);
preward = p(2);
pswitch = p(3);
stick = p(4);
eps=0.00001;

Q = [.5 .5];

if nargin>2
    
    llh = priorp(p,prior);
else
llh=0;
end

for t=1:size(beh,1)
    W = Q;
    if t>1
        W(a) = W(a)+stick;
    end
    
    a = beh(t,2);
    sftmx=1/(1+exp(beta*(W(3-a)-W(a))));
    llh = llh - log(sftmx);
    
    r = beh(t,3);
    
    if r==1
        likelihood(3-a) = 1-preward;
        likelihood(a) = preward;
    else
        likelihood(3-a) = preward;
        likelihood(a) = 1-preward;
    end
    Q = Q.*likelihood;
    Q = Q/sum(Q);
    Q = (1-pswitch)*Q + pswitch*(1-Q);
    
        
end
end

function llp = priorp(p,prior)
llp=0;

for i=1:length(p)
    Pr = prior{i};
    [~,T]=min(abs(Pr(1,:)-p(i)));
    llp = llp-log(max(Pr(2,T)));
end

end