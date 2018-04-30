import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from ac_agent import *
from not_dqn import *

# length of board
#L = 10
#boat_shapes = [(2,4), (1,5), (1,3), (1,3), (1,3)]

L = 4
boat_shapes = [(1,4), (1,3)]

# L = 6
# boat_shapes = [(2,4), (1,5), (1,3)]

# L = 3
# boat_shapes = [(2,2)]

def get_board():
  total_mass = sum([x[0]*x[1] for x in boat_shapes])
  def _gen_boats():
    ret = np.zeros([L, L])
    occupied = []
    poses = []

    joint_cstr = []
    for b_sh in boat_shapes:
      crd = np.random.randint(0, L-1, [2])
      wh,d = rand_orient(*b_sh)
      joint_cstr.append(rect_constr(crd, wh))
      poses.append((crd[0],crd[1],d))

    joint_constr = or_constr(joint_cstr)
    for y in range(L):
      for x in range(L):
        if joint_constr((x,y)):
          occupied.append((x,y))
          ret[y][x] = 1

    return ret, set(occupied), poses

  ret, occupied, poses = _gen_boats()
  if len(occupied) == total_mass:
    return ret, occupied, poses
  else:
    return get_board()

def rand_orient(w,h):
  if np.random.random() < 0.5:
    return (w,h),True
  else:
    return (h,w),False

def rect_constr(left_top, wid_hei):
  left, top = left_top
  wid, hei = wid_hei
  right, down = left + wid, top+hei
  def constr(crd):
    xx, yy = crd
    in_e1 = xx >= left
    in_e2 = xx < right
    in_e3 = yy >= top
    in_e4 = yy < down
    return in_e1 and in_e2 and in_e3 and in_e4
  return constr

def or_constr(crs):
  def constr(crd):
    for cr in crs:
      if cr(crd):
        return True
    return False
  return constr

def mask_board(board, made_moves):
  # return board

  board = np.copy(board)
  for x in range(L):
    for y in range(L):
      if (x,y) not in made_moves:
        board[y][x] = 2
  return board

class GameEnv(object):

  def __init__(self):
    self.board, self.occupied, _ = get_board()
    self.possible_actions = list(range(L*L))

  def win(self):
    return self.occupied.issubset(self.made_moves)

  def reset(self):
    # print('Start game')
    self.made_moves = set()
    return mask_board(self.board, self.made_moves), self.board

  def forbid(self):
    ret = set()
    for x,y in self.made_moves:
      ret.add( x * L + y )
    return ret

  def get_reward(self, x, y):
    if (self.board[y][x] == 1 and (x,y) not in self.made_moves):
      return 1.0
    # return 1.0
    if (x,y) in self.made_moves:
      return -1.0
    return -1.0

  # def get_final_reward(self):
  #   return len(self.occupied.intersection(self.made_moves))

  def step(self, action):
    x, y = action // L, action % L
    reward = self.get_reward(x,y)
    self.made_moves.add((x,y))
    done = self.win()
    # if done:
    #   reward = self.get_final_reward()
    state = mask_board(self.board, self.made_moves), self.board
    return state, reward, done

class StateXform:
  def __init__(self):
    self.length = L*L*2 * 2
  def board_to_np(self, state):
    ret = np.zeros(shape=(L*L,2), dtype=np.float32)
    ret_idx = np.resize(state, L*L)
    for i in range(L*L):
      if int(ret_idx[i]) != 2:
        ret[i, int(ret_idx[i])] = 1.0
    ret = np.resize(ret, L*L*2)
    return ret
  def state_to_np(self, state):
    board_mask, board_truth = state
    ret =  np.concatenate((self.board_to_np(board_mask),\
                           self.board_to_np(board_truth)))
    # ret =  self.board_to_np(board_mask)
    # ret =  self.board_to_np(board_mask)
    # ret =  self.board_to_np(board_truth)
    return ret

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(L*L))
    self.length = L*L
  def idx_to_action(self, idx):
    return self.possible_actions[idx]
  def action_to_idx(self, a):
    return a
  def action_to_1hot(self, a):
    ret = np.zeros(L*L)
    ret[a] = 1.0
    return ret

def measure(agent, game_bound):
  score = 0.0
  for i in range(100):
    env = GameEnv()
    trace = play_game(env, agent, game_bound, det=True)
    score += sum([tr.r for tr in trace])

  print ("# # # a deterministic trace # # # ")
  for tr in trace:
    print(tr.s)
    print(tr.a, tr.r)
    print(tr.v)
  return score / 100

def run_policy_gradient():
  # r_actor = RandomActor(env.possible_actions)
  state_xform, action_xform = StateXform(), ActionXform()
  ac_actor = PGAgent(state_xform, action_xform).cuda()
  buff = Buffer(10000)
  game_bound = L*L*0.75

  for i in range(1000000):
    if i % 1000 == 0:
      ac_actor.explore *= 0.95
      print ("explor rate ", ac_actor.explore)
      print (" ================= MEASURE  :  ", measure(ac_actor, game_bound))

    env = GameEnv()
    trace = play_game(env, ac_actor, game_bound)
    disc_trace = get_discount_trace(trace, ac_actor.value_estimator)
    [buff.add(tr) for tr in disc_trace]
    tr_sample = [buff.sample() for _ in range(50)]
    ac_actor.learn(tr_sample)

def run_table_q():
  action_xform = ActionXform()
  q_actor = TDLearn(action_xform)
  buff = Buffer(10000)
  game_bound = L*L*0.75

  for i in range(1000000):
    if i % 100 == 0:
      print (" ================= MEASURE  :  ", measure(q_actor, game_bound))
      print (" state size ", len(q_actor.Q))
      # print (" everything ? ", q_actor.Q)
    env = GameEnv()
    trace = play_game(env, q_actor, game_bound)
    [buff.add(tr) for tr in trace]
    tr_sample = [buff.sample() for _ in range(50)]
    q_actor.learn(tr_sample)

if __name__ == "__main__":
  run_table_q()

