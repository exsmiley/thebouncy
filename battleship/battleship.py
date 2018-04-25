import numpy as np

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from ac_agent import *

# length of board
L = 10
boat_shapes = [(2,4), (1,5), (1,3), (1,3), (1,3)]

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
    return mask_board(self.board, self.made_moves)

  def step(self, action):
    x, y = action // L, action % L
    reward = 1.0 if (self.board[y][x] == 1 and (x,y) not in self.made_moves) else 0.0
    self.made_moves.add((x,y))
    done = self.win()
    state = mask_board(self.board, self.made_moves)
    return state, reward, done

class StateXform:
  def __init__(self):
    self.length = L*L*3
  def state_to_np(self, state):
    ret = np.zeros(shape=(L*L, 3), dtype=np.float32)
    ret_idx = np.resize(state, L*L)
    for i in range(L*L):
      ret[i, int(ret_idx[i])] = 1.0
    ret = np.resize(ret, L*L*3)
    return ret

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(L*L))
    self.length = L*L
  def idx_to_action(self, idx):
    return self.possible_actions[idx]

if __name__ == "__main__":
  # r_actor = RandomActor(env.possible_actions)
  state_xform, action_xform = StateXform(), ActionXform()
  ac_actor = ACAgent(state_xform, action_xform).cuda()
  ar_actor = RandomActor(action_xform.possible_actions)
  buff = Buffer(10000)

  for i in range(1000):
    env = GameEnv()
    trace = play_game(env, ac_actor, 1000)

    for tr in trace:
      # print (tr[0])
      # print (tr[1])
      print (tr[3])
      buff.add(tr)

    assert 0, "game over man"
    tr_sample = buff.sample()
    print (tr_sample)

