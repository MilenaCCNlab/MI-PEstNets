import numpy as np
import pandas as pd
from scipy import special
import random
from numpy.random import shuffle

###### RL Models ######

def HRLtask_HRL(parameters,numtrials,pval,pswitch,numbandits,agentid,minswitch):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      numbandits : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)
      minswitch : minimum number of trials required for the correct cues/arrows/stimuli to reverse (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """
  softmaxbeta = parameters[0] # softmax beta
  learningrate = parameters[1] # learning rate
  stickiness = parameters[2]
  epsilon = parameters[3]


  Q = 1/numbandits*np.ones([1,numbandits])[0] # initialize action values
  iter =0 # intialize the counter until the correct cue/arrow/stimulus switch
  cb = 1

  # equal representation of directions in 3 cues/arrows/stimuli across the trials
  a=np.array([[0,1,0],[1,0,1],[0,0,1],[0,1,1],[1,0,0]])
  a = np.repeat(a,numtrials/len(a),axis=0)
  shuffle(a)


  allrewards =[] #initialize list that will store all rewards
  allcorrectcues =[] #initialize list that will store all correct cues
  alltrials =[] #initialize list that will store all trials
  alliscorrectcue =[] #initialize list that will store whether an agent chose a correct cue
  alliters=[] # intialize the list that will store all counters until correct cue/arrow/stimulus switch
  allindexofselcolor=[] #initialize list that will store the index of the selected cue
  allchosenside =[] #initialize list that will store which action/side agent chose (1/0)
  isswitch = [0] * numtrials #initialize list that will store whether the switch occurred
  allstims0 = [] #initialize list that will store direction of first arrow
  allstims1 = [] #initialize list that will store direction of second arrow
  allstims2 = [] #initialize list that will store direction of third arrow


  for i in range(numtrials):
    # three stimuli, each pointing to left/right (where left == 1; right == 0)
    stim = a[i]
    # Q values are the Q values of cues
    W=Q
    if i > 0:
      W[b] = W[b]+stickiness

    # select the action using softmax
    sftmx_p = special.softmax(softmaxbeta * W) # generate the action probability using the softmax
    b = np.random.choice(numbandits, p = sftmx_p) # select the action using the probability

    # s= side the selected cue/arrow/stimulus is pointing to
    s = stim[b]

    # if there's noise add a possible slip
    if np.random.uniform(0,1,1)[0]<epsilon:
      s=1-s

    # check if the side of the selected cue is the same as the side of the correct cue
    cor = int(s==stim[cb])

    # reward with p probability
    r = int(np.random.uniform(0,1,1)[0] < pval[cor])

    # update the q value of the selected cue/arrow/stimulus
    Q[b]+=learningrate*(r-Q[b])

    # update the q value of other cues/arrows/stimuli (counterfactual learning)
    others = [x for x in list(np.arange(numbandits)) if x!=b ]
    Q[np.array(others)] += learningrate*((1-r)-Q[np.array(others)])

    # after n trials and satisfied pswitch criteria, switch the correct cue/arrow/stimulus
    if (iter>minswitch) and (np.random.uniform(0,1,1)[0]<pswitch):
      iter=1
      bs = np.array([x for x in list(np.arange(numbandits)) if x!=cb])
      cb = bs[random.choice([0,1])]
      if i<numtrials-1:
        isswitch[i+1]=1
    else:
      iter += 1

    #store all trial variables
    alltrials.append(i)
    allcorrectcues.append(cb)
    alliters.append(iter)
    allindexofselcolor.append(b)
    allchosenside.append(s)
    alliscorrectcue.append(cor)
    allrewards.append(r) #store
    allstims0.append(stim[0])
    allstims1.append(stim[1])
    allstims2.append(stim[2])
  data = pd.DataFrame({"agentid" : [agentid] * len(alltrials),
                         'correctcue' : allcorrectcues,
                         'rewards': allrewards,
                         'isswitch': isswitch,
                         'iscorrectcue': alliscorrectcue,
                         'trials':alltrials,
                         'chosenside':allchosenside,
                         'chosencue':allindexofselcolor,
                         'correctruleiteration':alliters,
                         'alpha': [learningrate]*len(alltrials),
                         'stickiness': [stickiness]*len(alltrials),
                         'allstims0':allstims0,
                         'allstims1':allstims1,
                         'allstims2':allstims2,
                         'beta':[softmaxbeta]*len(alltrials)})

  return data



#### Bayesian Models ####

def HRLtask_Bayes(parameters, numtrials, pval, pswitch, numbandits, agentid,minswitch):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      numbandits : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)
      minswitch : minimum number of trials required for the correct cues/arrows/stimuli to reverse (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """

  softmaxbeta = parameters[0] # softmax beta
  epsilon = parameters[1]

  iter =0
  cb = 1
  prior = 1/numbandits*np.ones([1,numbandits])[0]

  # equal representation of directions in 3 cues/arrows/stimuli across the trials

  a = np.array([[0,1,0],[1,0,1],[0,0,1],[0,1,1],[1,0,0]])
  a = np.repeat(a,numtrials/len(a),axis=0)
  shuffle(a)


  allrewards =[] #initialize list that will store all rewards
  allcorrectcues =[] #initialize list that will store all correct cues
  alltrials =[] #initialize list that will store all trials
  alliscorrectcue =[] #initialize list that will store whether an agent chose a correct cue
  alliters=[] # intialize the list that will store all counters until correct cue/arrow/stimulus switch
  allindexofselcolor=[] #initialize list that will store the index of the selected cue
  allchosenside =[] #initialize list that will store which action/side agent chose (1/0)
  isswitch = [0] * numtrials #initialize list that will store whether the switch occurred
  allstims0 = [] #initialize list that will store direction of first arrow
  allstims1 = [] #initialize list that will store direction of second arrow
  allstims2 = [] #initialize list that will store direction of third arrow
  allp =[] # initialize list that will store all p values for posterior/prior
  likelihood = np.nan*np.ones([1,numbandits])[0] # intialize likelihood

  for i in range(numtrials):
    # three stimuli, each pointing to left/right (where left == 1; right == 0)
    stim = a[i]
    sftmx_p = special.softmax(softmaxbeta * prior) # generate the cue/arrow/stimulus probability using the softmax
    b = np.random.choice(numbandits, p = sftmx_p) # select the cue/arrow/stimulus using the probability

    # s= side the selected cue/arrow/stimulus is pointing to (0 or 1)
    s = stim[b]
    # if there's noise add a possible slip
    if np.random.uniform(0,1,1)[0]<epsilon:
      s=1-s

    cor = int(s==stim[cb])
    r = int(np.random.uniform(0,1,1)[0] < pval[cor])

    for n in range(numbandits):
      likelihood[n]=pval[stim[n]==s]

    if r == 0:
      likelihood = 1-likelihood

    posterior=likelihood*prior
    p = posterior/np.sum(posterior)
    prior = (1-pswitch)*p+pswitch*(1-p)/np.sum(1-p)

    # after n trials and satisfied pswitch criteria, switch the correct cue/arrow/stimulus
    if (iter>minswitch) and (np.random.uniform(0,1,1)[0]<pswitch):
      iter=1
      bs = np.array([x for x in list(np.arange(numbandits)) if x!=cb])
      cb = bs[random.choice([0,1])]
      if i<numtrials-1:
        isswitch[i+1]=1
    else:
      iter += 1

    # store all trial variables
    alltrials.append(i)
    allcorrectcues.append(cb)
    alliters.append(iter)
    allindexofselcolor.append(b)
    allchosenside.append(s)
    alliscorrectcue.append(cor)
    allrewards.append(r)
    allp.append(p)
    allstims0.append(stim[0])
    allstims1.append(stim[1])
    allstims2.append(stim[2])

  data = pd.DataFrame({"agentid" : [agentid] * len(alltrials),
                         'correctcue' : allcorrectcues,
                         'rewards': allrewards,
                         'isswitch': isswitch,
                         'iscorrectcue': alliscorrectcue,
                         'trials':alltrials,
                         'chosenside':allchosenside,
                         'chosencue':allindexofselcolor,
                         'correctruleiteration':alliters,
                         'allp':allp,
                         'beta': [softmaxbeta]*len(alltrials),
                         'epsilon': [epsilon]*len(alltrials),
                         'allstims0':allstims0,
                         'allstims1':allstims1,
                         'allstims2':allstims2})

  return data





def HRLtask_StickyBayes(parameters,numtrials,pval,pswitch,numbandits,agentid,minswitch):
  """
    #### Inputs ####
      parameters : model parameter values (list)
      numtrials : number of trials we want to simulate for an agent (int)
      pval : probability of the correct action being rewarded (float)
      numbandits : total number of possible actions (int)
      agentid : the sequential ID label for the agent that is being simulated (int)
      minswitch : minimum number of trials required for the correct cues/arrows/stimuli to reverse (int)


    #### Outputs ####

      data : a pandas data frame containing true parameter values, agent actions/rewards, agent id, trials etc.
  """

  softmaxbeta = parameters[0] # softmax beta
  epsilon = parameters[1]
  stick = parameters[2]

  iter =0
  cb = 1
  prior = 1/numbandits*np.ones([1,numbandits])[0]

  # equal representation of directions in 3 cues/arrows/stimuli across the trials

  a=np.array([[0,1,0],[1,0,1],[0,0,1],[0,1,1],[1,0,0]])
  a = np.repeat(a,numtrials/len(a),axis=0)
  shuffle(a)


  allrewards =[] #initialize list that will store all rewards
  allcorrectcues =[] #initialize list that will store all correct cues
  alltrials =[] #initialize list that will store all trials
  alliscorrectcue =[] #initialize list that will store whether an agent chose a correct cue
  alliters=[] # intialize the list that will store all counters until correct cue/arrow/stimulus switch
  allindexofselcolor=[] #initialize list that will store the index of the selected cue
  allchosenside =[] #initialize list that will store which action/side agent chose (1/0)
  isswitch = [0] * numtrials #initialize list that will store whether the switch occurred
  allstims0 = [] #initialize list that will store direction of first arrow
  allstims1 = [] #initialize list that will store direction of second arrow
  allstims2 = [] #initialize list that will store direction of third arrow
  allp =[] # initialize list that will store all p values for posterior/prior
  likelihood = np.nan*np.ones([1,numbandits])[0] # intialize likelihood




  for i in range(numtrials):
    # three stimuli, each pointing to left/right (where left == 1; right == 0)
    stim = a[i]
    W = np.log(prior)

    if i >0:
       W[b]=W[b]+stick

    sftmx_p = special.softmax(softmaxbeta * W) # generate the cue/arrow/stimulus probability using the softmax
    b = np.random.choice(numbandits, p = sftmx_p) # select the cue/arrow/stimulus using the probability
    # s= side the selected cue/arrow/stimulus is pointing to (0 or 1)
    s = stim[b]
    # if there's noise add a possible slip
    if np.random.uniform(0,1,1)[0]<epsilon:
      s=1-s

    cor = int(s==stim[cb])
    r = int(np.random.uniform(0,1,1)[0] < pval[cor])

    for n in range(numbandits):
      likelihood[n]=pval[stim[n]==s]

    if r == 0:
      likelihood = 1-likelihood

    posterior=likelihood*prior
    p = posterior/np.sum(posterior)
    prior = (1-pswitch)*p+pswitch*(1-p)/np.sum(1-p)


    if (iter>minswitch) and (np.random.uniform(0,1,1)[0]<pswitch):
      iter=1
      bs = np.array([x for x in list(np.arange(numbandits)) if x!=cb])
      cb = bs[random.choice([0,1])]
      if i<numtrials-1:
        isswitch[i+1]=1
    else:
      iter += 1

    # store all trial variables
    alltrials.append(i)
    allcorrectcues.append(cb)
    alliters.append(iter)
    allindexofselcolor.append(b)
    allchosenside.append(s)
    alliscorrectcue.append(cor)
    allrewards.append(r)
    allp.append(p)
    allstims0.append(stim[0])
    allstims1.append(stim[1])
    allstims2.append(stim[2])

  data = pd.DataFrame({"agentid" : [agentid] * len(alltrials),
                         'correctcue' : allcorrectcues,
                         'rewards': allrewards,
                         'isswitch': isswitch,
                         'iscorrectcue': alliscorrectcue,
                         'trials':alltrials,
                         'chosenside':allchosenside,
                         'chosencue':allindexofselcolor,
                         'correctruleiteration':alliters,
                         'allp':allp,
                         'beta': [softmaxbeta]*len(alltrials),
                         'epsilon': [epsilon]*len(alltrials),
                         'stickiness':[stick]*len(alltrials),
                         'allstims0':allstims0,
                         'allstims1':allstims1,
                         'allstims2':allstims2})

  return data

