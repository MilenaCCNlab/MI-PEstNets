clear;
close all;
%% choose fitting method

whichMod = 4;
isMAP = 0;


if isMAP == 1
    fitmethod = 'MAP';
else
    fitmethod = 'MLE';
end
%% model specification
models = [];
model_names= ['RL2p','RL4p','Bayes','StickyBayes'];


if whichMod == 1

    curr_model = [];
    curr_model.name = 'RL2p';
    curr_model.name_save ='2p_RL';
    curr_model.data_param_name = '2ParamRL';
    curr_model.pMin = [0 0 ];
    curr_model.pMax = [1 1 ];
    curr_model.pnames = {'beta','alpha'};
    curr_model.capacity = 0;
    curr_model.mult = [10,1];
    curr_model.sim_min = [0.2 0.5];
    curr_model.sim_max = [0.6 1];
    models{1}=curr_model;


elseif whichMod == 2

    curr_model = [];
    curr_model.name = 'RL4p';
    curr_model.name_save ='4p_RL';
    curr_model.data_param_name = '4ParamRL';
    curr_model.pMin = [0 0 0 -1];
    curr_model.pMax = [1 1 1 1];
    curr_model.pnames = {'beta','neg_alpha','alpha','stickiness'};

    curr_model.mult =    [10, 1,  1,    1];
    curr_model.sim_min = [0.2 0.5 0.5 -.1];
    curr_model.sim_max = [0.6 1   1    .4];

    models{1}=curr_model;


elseif whichMod == 3

    curr_model = [];
    curr_model.name = 'Bayes';
    curr_model.name_save ='Bayes';
    curr_model.data_param_name = 'Bayes';
    curr_model.pMin = [0 0 0 ];
    curr_model.pMax = [1 1 1 ];
    curr_model.pnames = {'beta','preward','pswitch'};
    curr_model.capacity = 0;
    curr_model.mult = [10,1,1];
    curr_model.sim_min = [0.2 0.7 0.01];
    curr_model.sim_max = [0.6 1 0.3];


elseif whichMod ==4

    curr_model = [];
    curr_model.name = 'StickyBayes';
    curr_model.name_save ='StickyBayes';
    curr_model.data_param_name = 'StickyBayes';
    curr_model.pMin = [0 0 0 -1];
    curr_model.pMax = [1 1 1 1];
    curr_model.pnames = {'beta','preward','pswitch','stickiness'};
    curr_model.capacity = 0;
    curr_model.mult = [10,1,1 1];
    curr_model.sim_min = [0.2 0.7 0.01 -.1];
    curr_model.sim_max = [0.6 1 0.3 .4];

end
%% load data


agents = eval(['readtable("3000agent_500t_',curr_model.data_param_name,'_test.csv")']);
numfitmodels = 1;
models{numfitmodels}=curr_model;

%% task parameters

NT = 500;%number of trials
minswitch=10;% number of trials before switch
th = 0.8;% threshold: probability or reward|correct

%% estimation parameters

num_iter = 20; % number of starting points to test
options = optimoptions('fmincon','Display','off');

%% sample simulating parameters, and "empirical" prior for MAP

agents.agentid = agents.agentid+1;
allagents= unique(agents.agentid);
nsim = length(allagents);
curr_model = models{1};
for ms=1:numfitmodels
    sim_model = models{ms};
    np = length(sim_model.pMin);
    p_sims{ms} = sim_model.sim_min + rand(nsim,np).*(sim_model.sim_max-sim_model.sim_min);
    priors{ms} = computeprior(p_sims{ms},curr_model.pnames);
end


%% do generate and recover of models and parameters

ms=1;
mf=1;
for s =1:nsim

    disp(num2str(s))

    beh = agents(agents.agentid==s,:);
    param_idx_agent = [];
    % order and grab parameters from the dataframe
    for pidx = 1:length(curr_model.pnames)
        param_idx_agent = [param_idx_agent,find(ismember(beh.Properties.VariableNames,curr_model.pnames{pidx})==1)];

    end
    % extract true parameters data was simulted from
    trueparameters = beh(:,param_idx_agent);
    trueparameters = trueparameters(1,:);
    % agent behavior we are fitting: selected actions,rewards,correct
    % actions
    beh = [beh.correct_actions,beh.actions,beh.rewards];
    beh(:,1)=beh(:,1)+1;
    beh(:,2)=beh(:,2)+1;
    prior = priors{1};


    % pass appropriate arguments depending on whether we are doing MAP/MLE
    if isMAP == 1
        eval(['myfitfun = @(par) llh',curr_model.name,'(par,beh,prior);']);
    else
        eval(['myfitfun = @(par) llh',curr_model.name,'(par,beh);']);
    end


    j=1;
    likelihoods = [];
    p_tested=[];
    
    
    for n = 1:num_iter
        % set random starting parameters to get out of local minima
        par = curr_model.pMin+rand(1,np).*(curr_model.pMax-curr_model.pMin);

        % call fmincon, passing function handle, parameter start values, and constraints
        [p,fval,exitflag,output,lambda,grad,hessian] = ...
            fmincon(myfitfun,par,[],[],[],[],curr_model.pMin,...
            curr_model.pMax,[],options);
        % store estimated parameters on each iterations
        p_tested(j,:) = p;
        % store likelihoods on each iteration
        likelihoods(j) = fval;

        % update counter
        j=j+1;
    end

    % save true parameters, best parameters and corresponding LLH for each subject
    bestind = find(likelihoods == min(likelihoods));
    if length(bestind) > 1, bestind = bestind(randi(1)); end
    best_params(s,:) = p_tested(bestind,:);
    best_likelihoods(s) = -likelihoods(bestind);
    true_params(s,:) = table2array(trueparameters);

end


% concatenate estimated and true parameters and assign parameter labels

best_params = best_params.*curr_model.mult;
all_params = [best_params,true_params];
param_labels = {};
true_param_labels = {};
for np = 1:length(curr_model.pnames)
    if isMAP == 1
        param_labels{np} = ['map_',curr_model.pnames{np}];
    else
        param_labels{np} = ['mle_',curr_model.pnames{np}];
    end
    true_param_labels{np}= ['true_',curr_model.pnames{np}];

end

all_paramlabels=[param_labels,true_param_labels];
all_params=array2table(all_params);
all_params.Properties.VariableNames = all_paramlabels;

% visualize parameter recovery
x = all_params.true_preward;
y = all_params.mle_preward;

figure
hold on;

scatter(x,y)
lsline
plot(x,x,LineWidth=3,Color='k')

keyboard;
% store estimates   
eval(['writetable(all_params,"',curr_model.name_save,'_',fitmethod,'_estimates.csv")']);

