from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ACAgent(nn.Module):
  '''
  g e n e r i c   a c t o r   c r i t i c   a g e n t
  
  '''

  def __init__(self, state_xform, action_xform):
    '''
    the state_xform and action_xform are little objects that can transform
    the state and action between the game representation and 1d array representation
    '''
    super(ACAgent, self).__init__()
    state_length, action_length = state_xform.length, action_xform.length
    self.state_xform, self.action_xform = state_xform, action_xform
    # 1 hidden layer, then predict the action and the value of state
    self.action_enc  = nn.Linear(state_length, state_length)
    self.action_pred = nn.Linear(state_length, action_length)
    self.value_enc  = nn.Linear(state_length, state_length)
    self.value_pred = nn.Linear(state_length, 1)

    self.actor_opt = torch.optim.Adam(\
    list(self.action_enc.parameters()) +
    list(self.action_pred.parameters()),
    lr = 0.001)

    self.critic_opt = torch.optim.Adam(\
    list(self.value_enc.parameters()) +
    list(self.value_pred.parameters()),
    lr = 0.1)

  def actor_forward(self, x):
    x = F.relu(self.action_enc(x))
    return x
  def critic_forward(self, x):
    x = F.relu(self.value_enc(x))
    return x

  # predict the probablity and the value on input state x
  # work over batched dimensions [batch x ...]
  def predict(self, x):
    action_scores = self.action_pred(self.actor_forward(x))
    action_pr = F.softmax(action_scores, dim=-1)
    state_value = self.value_pred(self.critic_forward(x))
    return action_pr, state_value

  # generate an action from a game state x
  def act(self, x, forbid=set([])):
    x = self.state_xform.state_to_np(x)
    # wrap an additional dimension around it so it has a "batch" dimension
    x = to_torch(np.expand_dims(x,0))
    action_pr, _ = self.predict(x)
    # print ("action prob ", action_pr)
    action_pr = action_pr.data.cpu().numpy()[0]

    for idx in forbid:
      action_pr[idx] = 0.0
    action_pr /= np.sum(action_pr)
    
    action_id = np.random.choice(range(len(action_pr)), p=action_pr)
    return self.action_xform.idx_to_action(action_id), action_pr

  # predict the value of a game state x
  def value_estimator(self, x):
    x = self.state_xform.state_to_np(x)
    # wrap an additional dimension around it so it has a "batch" dimension
    x = to_torch(np.expand_dims(x,0))
    _, v_estimate = self.predict(x)
    return v_estimate.data.cpu().numpy()[0][0]

  def learn(self, tups):
    '''
    a tuple is (s, a, ss, r, v)
    s: current state
    a: action taken
    ss: next state
    r: aggregated discounted future reward
    v: value esitmation of current state (produced by an old network's v function)
    '''
    states = [] # states traversed
    actions = [] # actions taken on those states
    disc_rs = [] # discounted future returns
    for tr in tups:
      states.append(self.state_xform.state_to_np(tr.s))
      actions.append(self.action_xform.action_to_idx(tr.a))
      disc_rs.append(np.array([tr.r]))

    states = to_torch(np.array(states))
    actions = to_torch_int(np.array(actions))
    actions = actions.view(-1,1)
    disc_rs = to_torch(np.array(disc_rs))

    # predict the action probability and the values
    action_probs, v_estimates = self.predict(states)

    log_action_probs = torch.log(action_probs)
    entropy = -(action_probs * log_action_probs).sum(1).mean()

    chosen_action_log_prob = torch.gather(log_action_probs, 1, actions)

    advantage = disc_rs - v_estimates
    action_gain = (chosen_action_log_prob * advantage).mean()
    value_loss = advantage.pow(2).mean()

    # print ("entropy")
    # print (entropy)
    # print ("action_gain")
    # print (action_gain)
    # print ("value loss")
    # print (value_loss)

    loss = value_loss - action_gain - entropy

    self.actor_opt.zero_grad()
    self.critic_opt.zero_grad()
    loss.backward()
    self.actor_opt.step()
    self.critic_opt.step()

  def load(self, loc):
    self.load_state_dict(torch.load(loc))
    print('Loaded model from {}!'.format(loc))

  def save(self, loc):
    torch.save(self.state_dict(), loc)
    print('Saved model to {}!'.format(loc))


class PGAgent(nn.Module):
  '''
  g e n e r i c   policy gradient
  
  '''

  def __init__(self, state_xform, action_xform):
    '''
    the state_xform and action_xform are little objects that can transform
    the state and action between the game representation and 1d array representation
    '''
    super(PGAgent, self).__init__()
    self.explore = 1.0

    state_length, action_length = state_xform.length, action_xform.length
    self.state_xform, self.action_xform = state_xform, action_xform
    # 1 hidden layer, then predict the action and the value of state
    self.enc = nn.Linear(state_length, state_length * 10)
    self.action_pred = nn.Linear(state_length * 10, action_length)
    self.all_opt = torch.optim.Adam(self.parameters(), lr=1e-3)

  # predict the probablity and the value on input state x
  # work over batched dimensions [batch x ...]
  def predict(self, x):
    x = F.relu(self.enc(x)) 
    action_scores = self.action_pred(x)
    # action_pr = F.softmax(action_scores, dim=-1) + 1e-8
    action_pr = F.softmax(action_scores, dim=-1) + (self.explore / self.action_xform.length)
    #sumsum = torch.sum(action_pr, dim=1).expand_as(action_pr)
    sumsum = torch.sum(action_pr, dim=1, keepdim=True)
    action_pr = action_pr.div(sumsum)
    state_value = 0.0
    return action_pr, state_value

  # generate an action from a game state x
  def act(self, x, forbid=set([]), det=False):
    x = self.state_xform.state_to_np(x)
    # wrap an additional dimension around it so it has a "batch" dimension
    x = to_torch(np.expand_dims(x,0))
    action_pr, _ = self.predict(x)
    # print ("action prob ", action_pr)
    action_pr = action_pr.data.cpu().numpy()[0]

    for idx in forbid:
      action_pr[idx] = 0.0
    action_pr /= np.sum(action_pr)

    action_id = np.random.choice(range(len(action_pr)), p=action_pr) \
                if not det \
                else np.argmax(action_pr)
    return self.action_xform.idx_to_action(action_id), action_pr

  # predict the value of a game state x
  def value_estimator(self, x):
    return 0.0

  def learn(self, tups):
    '''
    a tuple is (s, a, ss, r, v)
    s: current state
    a: action taken
    ss: next state
    r: aggregated discounted future reward
    v: value esitmation of current state (produced by an old network's v function)
    '''
    states = []
    target_actions = []
    disc_rs = []
    for tr in tups:
      states.append(self.state_xform.state_to_np(tr.s))
      target_actions.append(self.action_xform.action_to_1hot(tr.a))
      disc_rs.append(np.array([tr.r]))

    states = to_torch(np.array(states))
    target_actions = to_torch(np.array(target_actions))
    disc_rs = to_torch(np.array(disc_rs))
    
    # predict the action probability and the values
    action_preds, value_preds = self.predict(states)
    log_action_preds = torch.log(action_preds)
    action_xentropy = torch.sum(log_action_preds * target_actions, dim=1)
    weighted_action_xentropy = disc_rs * action_xentropy

    overall_fitness = torch.sum(weighted_action_xentropy)

    cost = - overall_fitness
    self.all_opt.zero_grad()
    cost.backward()
    nn.utils.clip_grad_norm(self.parameters(), 0.5)
    self.all_opt.step()

  def load(self, loc):
    self.load_state_dict(torch.load(loc))
    print('Loaded model from {}!'.format(loc))

  def save(self, loc):
    torch.save(self.state_dict(), loc)
    print('Saved model to {}!'.format(loc))
