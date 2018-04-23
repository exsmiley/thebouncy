import numpy as np
import random

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
    for i in range(L):
      for j in range(L):
        if joint_constr((i,j)):
          occupied.append((i,j))
          ret[i][j] = 1.0

    return ret, occupied, poses

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
    print ("okay init")
    self.possible_actions = list(range(L*L))

  def win(self):
    for ocu in self.occupied:
      if ocu not in self.made_moves:
        return False
    return True

  def reset(self):
    # print('Start game')
    self.board, self.occupied, _ = get_board()
    self.made_moves = set()
    return mask_board(self.board, self.made_moves)

  def step(self, action):
    x, y = action // L, action % L
    self.made_moves.add((x,y))
    reward = 1.0 if self.board[y][x] == 1 else 0.0
    done = self.win()
    state = mask_board(self.board, self.made_moves)
    return state, reward, done

class RandomActor:

  def __init__(self, possible_actions):
    self.possible_actions = possible_actions

  def act(self, state):
    return random.choice(self.possible_actions)

def play_game(env, actor):
  trace = []
  s = env.reset()
  done = False

  while not done:
    action = actor.act(s)
    ss, r, done = env.step(action)
    trace.append( (s, action, ss, r) )
    s = ss

  return trace

if __name__ == "__main__":
  env = GameEnv()
  r_actor = RandomActor(env.possible_actions)
  trace = play_game(env, r_actor)
  print (len(trace))

  print (trace[0])

  print ("JKFJDSKLFSDJLFSD")
  print (trace[-1])
