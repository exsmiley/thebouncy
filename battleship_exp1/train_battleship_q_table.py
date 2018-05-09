def run_policy_gradient():
  # r_actor = RandomActor(env.possible_actions)
  state_xform, action_xform = StateXform(), ActionXform()
  ac_actor = PGAgent(state_xform, action_xform).to(device)
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


