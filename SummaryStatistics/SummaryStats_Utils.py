
import numpy as np


def analyzeSwitch(subdata,numPreSwitchTrials,numPostSwitchTrials):

  switches= [k for k, x in enumerate(subdata['isswitch']) if x]

  corr =  np.array(subdata['iscorrect'])
  corr = corr.astype(int)
  switches=[x for x in switches if x <= subdata.shape[0]-numPostSwitchTrials]
  allswitchaccuracy=np.nan*np.ones([len(switches),numPreSwitchTrials+numPostSwitchTrials])

  for s in range(len(switches)):
    sw = switches[s]
    allswitchaccuracy[s]= np.array(corr[np.arange(sw-numPreSwitchTrials,sw+numPostSwitchTrials)])

  LC=np.nanmean(allswitchaccuracy,0)

  return LC




def three_back_analysis_HRL(subdata):

  reward = np.array(subdata.rewards)
  chosen_side = np.array(subdata.chosenside)
  stims = np.array([subdata.allstims0,subdata.allstims1,subdata.allstims2])
  p3 = np.zeros((len(reward)))
  outcomes = [0,1]
  k = 0

  r1=reward[:len(reward)-3]
  r2=reward[1:len(reward)-2]
  r3=reward[2:len(reward)-1]

  firstnanvals = np.array([np.nan,np.nan,np.nan])

  for t1 in outcomes:
    for t2 in outcomes:
      for t3 in outcomes:
        k +=1
        r1_1 = r1 == t3
        r2_1 = r2 == t2
        r3_1 = r3 == t1
        allcorr = r1_1 & r2_1 & r3_1
        allcorr1 = np.concatenate([firstnanvals,allcorr])

        p3 += (k*allcorr1)

  is_stay =[]
  for t in range(len(reward)-1):
    what_possible_chosen_cue = np.where(stims[:,t]==chosen_side[t])[0]
    what_possible_chosen_cue_nextt = np.where(stims[:,t+1]==chosen_side[t+1])[0]
    is_stay.append(np.any(what_possible_chosen_cue_nextt==what_possible_chosen_cue))

  is_stay = np.insert(is_stay,0,np.nan)
  stay_p3 =[]
  for i in range(1,9):
    stay_p3.append(np.nanmean(is_stay[p3==i]))


  return stay_p3




def previous_Cue(subdata,numPostSwitchTrials,numPreSwitchTrials,numtrials):

  correct_cue = np.array(subdata.correctcue)
  chosen_side = np.array(subdata.chosenside)
  ruleswitch = correct_cue[1:len(correct_cue)]==correct_cue[:len(correct_cue)-1]
  switches = np.where(ruleswitch==False)[0]
  stims = np.array([subdata.allstims0,subdata.allstims1,subdata.allstims2])
  chosen_cue=[]

  for t in range(numtrials):
    chosen_cue.append(np.where(stims[:,t]==chosen_side[t])[0])
  chosen_cue = np.array(chosen_cue)


  switches=[x for x in switches if x <= subdata.shape[0]-numPostSwitchTrials]

  allswitch_ispreviouscue=np.nan*np.ones([len(switches),numPreSwitchTrials+numPostSwitchTrials])
  for s in range(len(switches)):
    sw = switches[s]
    prev_correct=correct_cue[sw-1]
    is_previous_corr_cue=np.array([np.any(chosen_cue[m]==prev_correct) for m in range(len(chosen_cue))])

    allswitch_ispreviouscue[s] =np.array(is_previous_corr_cue[np.arange(sw-numPreSwitchTrials,sw+numPostSwitchTrials)])

  PC=np.nanmean(allswitch_ispreviouscue,0)
  return PC


def three_back_analysis_PRL(subdata):
  reward = np.array(subdata.rewards)
  p3 = np.array([0]*len(reward)).astype(float)
  outcomes = [0,1]
  actions = np.array(subdata.actions)
  k = 0
  r1=reward[:len(reward)-3]
  r2=reward[1:len(reward)-2]
  r3=reward[2:len(reward)-1]

  firstnanvals = np.array([np.nan,np.nan,np.nan])

  for t1 in outcomes:
    for t2 in outcomes:
      for t3 in outcomes:
        k +=1
        r1_1 = r1 == t3
        r2_1 = r2 == t2
        r3_1 = r3 == t1
        allcorr = r1_1 & r2_1 & r3_1
        allcorr1 = np.concatenate([firstnanvals,allcorr])

        p3 += (k*allcorr1)

  isStay = actions[:len(actions)-1]==actions[1:len(actions)].astype(float)
  isStay = np.insert(isStay,0,np.nan)
  stay_3p =[]
  for i in range(1,9):
    stay_3p.append(np.nanmean(isStay[p3==i]))

  return stay_3p



def abanalysis(subdata):
  actions =np.array(subdata.actions)
  reward = np.array(subdata.rewards)
  endidx = len(actions)

  isstay = actions[1:endidx]==actions[:endidx-1].astype(float)
  isstay =np.insert(isstay,0,np.nan)
  r1 = reward[0:endidx-2]
  r2 = reward[1:endidx-1]
  stay = isstay[1:endidx-1]



  p2 = np.array([0]*len(reward)).astype(float)
  k =0
  outcomes = [0,1]
  firstnanvals = np.array([np.nan,np.nan])
  for t2 in outcomes:
    for t1 in outcomes:
      for s1 in [1,0]:
        k += 1
        r1_1 = r1 == t2
        r2_1 = r2 == t1
        stay_2 = stay == s1
        allcorr = r1_1 & r2_1 & stay_2
        allcorr1 = np.concatenate([firstnanvals,allcorr])
        p2 += (k*allcorr1)
  stay2 = actions[:endidx-2]==actions[2:endidx].astype(float)
  stay2 = np.concatenate([firstnanvals,stay2])
  stayp2 =[]
  for i in range(1,9):
    stayp2.append(np.nanmean(stay2[p2==i]))

  return stayp2