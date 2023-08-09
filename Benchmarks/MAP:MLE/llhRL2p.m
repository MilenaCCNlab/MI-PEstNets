function llh = llhRL2p(p,beh, prior)

beta = 10*p(1);
alpha = p(2);
stick = 0;


Q = [.5 .5];
if nargin>2 % if we are doing MAP, passed 3 arguments (3rd is the prior)
    
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
    

    Q(a) = Q(a) + alpha*(r-Q(a));
    unch=3-a;
    Q(unch) = Q(unch) + alpha*((1-r)-Q(unch));
           
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