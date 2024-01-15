import numpy as np
import pandas as pd
from scipy import special
import random
import copy
###### RL Models ######

def PRLtask_2ParamRL(parameters,numtrials,pval,minswitch,numaction,agentid):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      minswitch : minimum number of trials required for the correct actions to reverse (int)
      numaction : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """

  softmaxbeta = parameters[0]*10 # softmax beta
  learningrate = parameters[1] # learning rate

  actionQvalues = np.array([1/numaction] * numaction) # initialize action Q values (1/number of actions)
  CurrentlyCorrect = random.choice([0,1]) # randomly initialize the action that is more likely to be rewarded at first
  currLength = minswitch+random.randint(0,5) # the number of correct trials required for the correct action to switch
  currCum = 0 # initialize cumulative reward

  allactions = [] # initialize list that will store all trial actions
  allrewards =[] # initialize list that will store all trial rewards
  allcorrectchoices =[] #initialize list that will store all trial correct actions
  isswitch= [0] * numtrials #initialize the list that will store an index of switch trials (1 if switch, 0 otherwise)
  alltrials = [] # initialize the list that will store the list of trials
  alliscorrectaction =[] # initialize the list that will store whether the agent selected the currently correct action or not (different from the reward)
  qvalA1 =[]
  qvalA2 =[]

  for i in range(numtrials):

    sftmx_p = special.softmax(softmaxbeta * actionQvalues) # generate the action probability using the softmax
    action = np.random.choice(numaction, p = sftmx_p) # select the action using the probability
    correct = (action == CurrentlyCorrect) # is the selected action the action that is currently rewarding
    correct=int(correct)
    genrandomvalue = np.random.uniform(0,1,1)[0]

    if genrandomvalue<pval: # if correct, reward with p probability
      r = correct
    else:
      r = 1-correct

    RPE = r - actionQvalues[action]
    actionQvalues[action] +=(learningrate*RPE) #update the action values based on the outcome



    unchosenaction=1-action # action that's not selected (for counterfactual learning)
    RPEunchosen = (1-r)-actionQvalues[unchosenaction] #RPE for the unselected action
    actionQvalues[unchosenaction] += (learningrate*RPEunchosen) # actionQvalues[unchosenaction] + # update the values of unselected actions

    currCum = currCum+r  # update cumulative reward
    if (r==1) and (currCum>=currLength): # check for the counter of the trials required to switch correct actions
      CurrentlyCorrect = 1-CurrentlyCorrect
      currLength = minswitch+random.randint(0,5)
      currCum=0
      if i < numtrials-1:
        isswitch[i+1]=1

    # store all trial variables
    allactions.append(action)
    allrewards.append(r)
    allcorrectchoices.append(CurrentlyCorrect) #store the action that was correct on the current trial
    alltrials.append(i)
    alliscorrectaction.append(correct)
    qvalA1.append(actionQvalues[0])
    qvalA2.append(actionQvalues[1])

  data = pd.DataFrame({"agentid" : [agentid] * len(allactions),
                         'actions': allactions,
                         'correct_actions' : allcorrectchoices,
                         'rewards': allrewards,
                         'isswitch': isswitch,
                         'iscorrectaction': alliscorrectaction,
                         'trials':alltrials,
                         'alpha': [learningrate]*len(allactions),
                         'beta': [softmaxbeta]*len(allrewards),
                         'Q_a1':qvalA1,
                         'Q_a2':qvalA2})

  return data




def PRLtask_4ParamRL(parameters,numtrials,pval,minswitch,numaction,agentid):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      minswitch : minimum number of trials required for the correct actions to reverse (int)
      numaction : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """
  softmaxbeta = 10*parameters[0] # softmax beta
  learningrate = parameters[1] # learning rate
  learningrateneg = parameters[2] # negative learning rate
  learningrates =[learningrateneg,learningrate]

  stickiness = parameters[3] # stickiness parameter

  actionQvalues =  np.array([1/numaction] * numaction) # initialize action values
  CurrentlyCorrect=random.choice([0,1]) # initialize the action that is more likely to be rewarded at first
  currLength = minswitch+random.randint(0,5) # the number of trials required for the correct action to switch
  currCum = 0 # initialize cumulative reward

  allactions = [] #initialize list that will store all actions
  allrewards =[] #initialize list that will store all rewards
  allcorrectchoices =[] #initialize list that will store all trial correct actions
  isswitch= [0] * numtrials #initialize the list that will store an index of switch trials (1 if switch, 0 otherwise)
  alltrials =[] # initialize the list that will store the list of trials
  alliscorrectaction =[] # initialize the list that will store whether the agent selected the currently correct action or not (different from the reward)

  for i in range(numtrials):

    W = copy.copy(actionQvalues)
    if i > 0:
      W[action] = W[action]+stickiness

    sftmx_p = special.softmax(softmaxbeta * W) # generate the action probability using the softmax
    action = np.random.choice(numaction, p = sftmx_p) # select the action using the probability
    correct = (action == CurrentlyCorrect) # is the selected action the action that is currently rewarding
    correct=int(correct)

    if np.random.uniform(0,1,1)[0]<pval: # generate the random value between 0 and 1, if it's smaller than set pvalue the reward is +1, otherwise 0
      r = correct
    else:
      r = 1-correct

    RPE = r - actionQvalues[action]
    unchosenaction=1-action # action that's not selected
    RPEunchosen = (1-r)-actionQvalues[unchosenaction] #RPE for the unselected action

    actionQvalues[action]+= (learningrates[r]*RPE) # update the action values based on the outcome
    actionQvalues[unchosenaction]+= (learningrates[r]*RPEunchosen)

    currCum = currCum+r  # update cumulative reward
    if (r==1) and (currCum>=currLength): # check for the counter of the trials required to switch correct actions
      CurrentlyCorrect = 1-CurrentlyCorrect
      currLength = minswitch+random.randint(0,5)
      currCum=0
      if i < numtrials-1:
        isswitch[i]=1

    # store all trial variables
    allactions.append(action)
    allrewards.append(r)
    allcorrectchoices.append(CurrentlyCorrect)
    alltrials.append(i)
    alliscorrectaction.append(correct)

  data = pd.DataFrame({"agentid" : [agentid] * len(allactions),
                         'actions': allactions,
                         'correct_actions' : allcorrectchoices,
                         'rewards': allrewards,
                         'isswitch': isswitch,
                         'iscorrectaction': alliscorrectaction,
                         'trials':alltrials,
                         'alpha': [learningrate]*len(allactions),
                         'beta': [softmaxbeta]*len(allrewards),
                         'neg_alpha': [learningrateneg]*len(allrewards),
                         'stickiness': [stickiness]*len(allrewards)})

  return data


def PRLtask_2ParamRL_attentionstate(parameters,numtrials,pval,minswitch,numaction,agentid):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      minswitch : minimum number of trials required for the correct actions to reverse (int)
      numaction : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """

  softmaxbeta = parameters[0]*10 # softmax beta
  learningrate = parameters[1] # learning rate
  T = parameters[2]
  tau = 1-(1/T)

  actionQvalues = np.array([1/numaction,1/numaction]) # initialize action values (1/number of actions)
  CurrentlyCorrect=random.choice([0,1]) # initialize the action that is more likely to be rewarded at first
  currLength = minswitch+random.randint(0,5) # the number of correct trials required for the correct action to switch
  currCum = 0 # initialize cumulative reward

  allactions = [] #initialize list that will store all actions
  allrewards =[] #initialize list that will store all rewards
  allcorrectchoices =[] #initialize list that will store all correct actions
  isswitch= [0] * numtrials #initialize the list that will store an index of switch trials (1 if switch, 0 otherwise)
  alltrials = [] # initialize the list that will store the list of trials
  alliscorrectaction =[] # initialize the list that will store whether the agent selected the currently correct action or not (different from the reward)
  qvalA1 =[]
  qvalA2 =[]
  all_whichstate = []
  whichState = 1 # initialize at attention state
  tau2  = 0.7


  for i in range(numtrials):

    if whichState == 0:
      if np.random.uniform(0,1,1)[0] > tau2:
        whichState = 1-whichState

    else:
      if np.random.uniform(0,1,1)[0]>tau:
        whichState = 1-whichState


    if whichState==1: # if in attention state
      sftmx_p = special.softmax(softmaxbeta * actionQvalues) # generate the action ps using the softmax
      action = np.random.choice(numaction, p = sftmx_p) # generate the action using the probability

    else: # if in random state

      action = random.choice([0, 1]) # choose randomly between the two actions


    correct = (action == CurrentlyCorrect) # is the selected action the action that is currently rewarding
    correct=int(correct)
    genrandomvalue = np.random.uniform(0,1,1)[0]

    if genrandomvalue<pval: # if correct, reward with p probability
      r = correct
    else:
      r = 1-correct

    RPE = r - actionQvalues[action]

    if whichState==1: # update if < smaller than tau
      actionQvalues[action] +=(learningrate*RPE)
      unchosenaction=1-action
      RPEunchosen = (1-r)-actionQvalues[unchosenaction]
      actionQvalues[unchosenaction] += (learningrate*RPEunchosen)


    currCum = currCum+r  # update cumulative reward
    if (r==1) and (currCum>=currLength): # check for the counter of the trials required to switch correct actions
      CurrentlyCorrect = 1-CurrentlyCorrect
      currLength = minswitch+random.randint(0,5)
      currCum=0
      if i < numtrials-1:
        isswitch[i+1]=1
    # store all trial variables
    allactions.append(action)
    allrewards.append(r)
    allcorrectchoices.append(CurrentlyCorrect)
    alltrials.append(i)
    alliscorrectaction.append(correct)
    qvalA1.append(actionQvalues[0])
    qvalA2.append(actionQvalues[1])
    all_whichstate.append(whichState)
  data = pd.DataFrame({"agentid" : [agentid] * len(allactions),
                         'actions': allactions,
                         'correct_actions' : allcorrectchoices,
                         'rewards': allrewards,
                         'isswitch': isswitch,
                         'iscorrectaction': alliscorrectaction,
                         'trials':alltrials,
                         'alpha': [learningrate]*len(allactions),
                         'beta': [softmaxbeta]*len(allrewards),
                         'Q_a1':qvalA1,
                         'Q_a2':qvalA2,
                         "tau":[tau]*len(allactions),
                         "T":[T]*len(allactions),
                         "which_state":all_whichstate})

  return data



###### Bayesian Models ######

def PRLtask_Bayes(parameters, numtrials, pval, minswitch, numbandits, agentid):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      minswitch : minimum number of trials required for the correct actions to reverse (int)
      numbandits : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """
  softmaxbeta = parameters[0] * 10  # softmax beta
  preward = parameters[1]
  pswitch = parameters[2]
  stick = parameters[3]

  Q = 1 / numbandits * np.ones([1, numbandits])[0]
  currCorr = random.choice([0, 1])
  currLength = minswitch + random.randint(0, 5)
  currCum = 0

  allactions = []  # initialize list that will store all actions
  allrewards = []  # initialize list that will store all rewards
  allcorrectchoices = []  # initialize list that will store all correct actions
  isswitch = [0] * numtrials  # initialize the list that will store an index of switch trials (1 if switch, 0 otherwise)
  alltrials = []  # initialize the list that will store the list of trials
  alliscorrectaction = []  # initialize the list that will store whether the agent selected the currently correct action or not (different from the reward)
  qvalA1 = []  # store all a1 q values
  qvalA2 = []  # store all a2 q values
  likelihood = np.nan * np.ones((1, 2))[0]  # initialize likelihood

  for i in range(numtrials):

    sftmx_p = special.softmax(softmaxbeta * Q)  # generate the action probability using the softmax
    a = np.random.choice(numbandits, p=sftmx_p)  # select the action using the probability

    cor = int(a == currCorr)
    genrandomvalue = np.random.uniform(0, 1, 1)[0]

    if genrandomvalue < pval:
      r = cor
    else:
      r = 1 - cor

    if r == 1:
      likelihood[1 - a] = 1 - preward
      likelihood[a] = preward
    else:
      likelihood[1 - a] = preward
      likelihood[a] = 1 - preward

    Q = Q * likelihood
    Q = Q / np.sum(Q)
    Q = ((1 - pswitch) * Q) + (pswitch * (1 - Q))
    currCum += r

    if (r == 1) and (currCum >= currLength):
      currCorr = 1 - currCorr
      currLength = minswitch + random.randint(0, 5)
      currCum = 0
      if i < numtrials - 1:
        isswitch[i + 1] = 1

    # store all trial variables
    allactions.append(a)
    allrewards.append(r)
    allcorrectchoices.append(currCorr)
    alltrials.append(i)
    alliscorrectaction.append(cor)
    qvalA1.append(Q[0])
    qvalA2.append(Q[1])

  data = pd.DataFrame({"agentid": [agentid] * len(allactions),
                       'actions': allactions,
                       'correct_actions': allcorrectchoices,
                       'rewards': allrewards,
                       'isswitch': isswitch,
                       'iscorrectaction': alliscorrectaction,
                       'trials': alltrials,
                       'preward': [preward] * len(allactions),
                       'beta': [softmaxbeta] * len(allrewards),
                       'pswitch': [pswitch] * len(allrewards),
                       'Q_a1': qvalA1,
                       'Q_a2': qvalA2})

  return data


def PRLtask_StickyBayes(parameters, numtrials, pval, minswitch, numbandits, agentid):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      minswitch : minimum number of trials required for the correct actions to reverse (int)
      numbandits : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """


  softmaxbeta = parameters[0] * 10  # softmax beta
  preward = parameters[1]
  pswitch = parameters[2]
  stick = parameters[3]

  Q = 1 / numbandits * np.ones([1, numbandits])[0]
  currCorr = random.choice([0, 1])
  currLength = minswitch + random.randint(0, 5)
  currCum = 0

  allactions = []  # initialize list that will store all actions
  allrewards = []  # initialize list that will store all rewards
  allcorrectchoices = []  # initialize list that will store all correct actions
  isswitch = [0] * numtrials  # initialize the list that will store an index of switch trials (1 if switch, 0 otherwise)
  alltrials = []  # initialize the list that will store the list of trials
  alliscorrectaction = []  # initialize the list that will store whether the agent selected the currently correct action or not (different from the reward)
  qvalA1 = [] # intialize list that will store action values
  qvalA2 = [] # intialize list that will store action values
  likelihood = np.nan * np.ones((1, 2))[0] # initialize likelihood

  for i in range(numtrials):

    W = copy.copy(Q)
    if i > 1:
      W[a] = W[a] + stick

    sftmx_p = special.softmax(softmaxbeta * W)  # generate the action probability using the softmax
    a = np.random.choice(numbandits, p=sftmx_p)  # select the action using the probability

    cor = int(a == currCorr)
    genrandomvalue = np.random.uniform(0, 1, 1)[0]

    if genrandomvalue < pval:
      r = cor
    else:
      r = 1 - cor

    if r == 1:
      likelihood[1 - a] = 1 - preward
      likelihood[a] = preward
    else:
      likelihood[1 - a] = preward
      likelihood[a] = 1 - preward

    Q = Q * likelihood
    Q = Q / np.sum(Q)
    Q = ((1 - pswitch) * Q) + (pswitch * (1 - Q))
    currCum += r

    if (r == 1) and (currCum >= currLength):
      currCorr = 1 - currCorr
      currLength = minswitch + random.randint(0, 5)
      currCum = 0
      if i < numtrials - 1:
        isswitch[i + 1] = 1

    allactions.append(a)  # store action
    allrewards.append(r)  # store
    allcorrectchoices.append(currCorr)  # store the action that was correct on the current trial
    alltrials.append(i)
    alliscorrectaction.append(cor)
    qvalA1.append(Q[0])
    qvalA2.append(Q[1])

  data = pd.DataFrame({"agentid": [agentid] * len(allactions),
                       'actions': allactions,
                       'correct_actions': allcorrectchoices,
                       'rewards': allrewards,
                       'isswitch': isswitch,
                       'iscorrectaction': alliscorrectaction,
                       'trials': alltrials,
                       'preward': [preward] * len(allactions),
                       'beta': [softmaxbeta] * len(allrewards),
                       'pswitch': [pswitch] * len(allrewards),
                       'stickiness': [stick] * len(allrewards),
                       'Q_a1': qvalA1,
                       'Q_a2': qvalA2})

  return data
