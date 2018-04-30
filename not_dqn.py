import random 

class Dict(dict):
  def __setitem__(self, key, value):
    key = repr(key)
    super(Dict, self).__setitem__(key, value)

  def __getitem__(self, key):
    key = repr(key)
    try:
      return super(Dict, self).__getitem__(key)
    except:
      return 0.0

class TDLearn:

  def __init__(self, action_xform):
    self.actions = action_xform.possible_actions 
    self.learn_rate = 0.1
    self.explore_rate = 0.1
    self.discount = 0.9
    self.Q = Dict()
    print ("table initialized ! " )

  def get_action_score_from_s(self, state):
    scores = []
    for a in self.actions:
      s,a = state, a
      scores.append((self.Q[(s,a)], a))
    return scores
  
  def act(self, state, forbid=set([]), det=False):
    scores = self.get_action_score_from_s(state)
    just_scores = [x[0] for x in scores]
    if det: 
      return max(scores)[1], just_scores
    else:
      if random.random() < self.explore_rate:  
        random_action = random.choice(scores) 
        return random_action[1], just_scores
      else:
        return max(scores)[1], just_scores

  def get_best_v(self, s):
    scores = self.get_action_score_from_s(s)
    return max(scores)[0]

  # predict the value of a game state x
  def value_estimator(self, x):
    return 0.0
    
  # trace is organized as a list of quadruples (s,a,s',r)
  # learn from SARSS
  def learn(self, trace):
    for step in trace:
      s,a,ss,r,_ = step
      best_future_v = self.get_best_v(ss)
      td = (r + self.discount * best_future_v) - self.Q[(s,a)]
      self.Q[(s,a)] = self.learn_rate * td + self.Q[(s,a)]

