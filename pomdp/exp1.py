from pomdp import *

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ac_agent import *

class StateXform:
  def __init__(self):
    self.length = 10
  def state_to_np(self, state):
    ret = np.zeros(shape=(10), dtype=np.float32)
    ret[state] = 1.0
    return ret

class ActionXform:
  def __init__(self):
    self.length = 2
  def idx_to_action(self, idx):
    return idx
  def action_to_idx(self, a):
    return a
  def action_to_1hot(self, a):
    return np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0])

if __name__ == "__main__":
  state_xform, action_xform = StateXform(), ActionXform()
  # ac_actor = PGAgent(state_xform, action_xform).cuda()
  ac_actor = ACAgent(state_xform, action_xform).cuda()
  buff = Buffer(1000)
  game_bound = 15
  env = RingEnv()

#  for i in range(100):
#    trace = play_game(env, ac_actor, game_bound)
#    disc_trace = get_discount_trace(trace, lambda x: x)
#    [buff.add(tr) for tr in disc_trace]

  for i in range(10000):
    trace = play_game(env, ac_actor, game_bound)
    print (trace[0].s, [(tr.s, tr.a, tr.v, tr.ss) for tr in trace], trace[-1].r)
    disc_trace = get_discount_trace(trace, lambda x: x)
    [buff.add(tr) for tr in disc_trace]

    tr_sample = [buff.sample() for _ in range(50)]
    ac_actor.learn(tr_sample)

