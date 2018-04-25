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
    self.enc1  = nn.Linear(state_length, state_length)
    self.enc2  = nn.Linear(state_length, state_length)
    self.action_pred = nn.Linear(state_length, action_length)
    self.value_pred = nn.Linear(state_length, 1)

  # predict the probablity and the value on input state x
  # work over batched dimensions [batch x ...]
  def predict(self, x):
    x = F.relu(self.enc2(self.enc1(x)))
    action_scores = self.action_pred(x)
    action_pr = F.softmax(action_scores, dim=-1) + 1e-8
    state_value = self.value_pred(x)
    return action_pr, state_value

  # generate an action from a game state x
  def act(self, x):
    x = self.state_xform.state_to_np(x)
    # wrap an additional dimension around it so it has a "batch" dimension
    x = to_torch(np.expand_dims(x,0))
    action_pr, _ = self.predict(x)
    action_pr = action_pr.data.cpu().numpy()[0]
    action_id = np.random.choice(range(len(action_pr)), p=action_pr)
    return self.action_xform.idx_to_action(action_id)

  def load(self, loc):
    self.load_state_dict(torch.load(loc))
    print('Loaded model from {}!'.format(loc))

  def save(self, loc):
    torch.save(self.state_dict(), loc)
    print('Saved model to {}!'.format(loc))


