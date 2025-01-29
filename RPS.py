def player(
    prev_play,
    opponent_history=[],
    q_table={},
    last_state=[("", "")],
    last_action=["R"],
    alpha=0.7,
    gamma=0.8,
    epsilon=[0.05]
):
    import random

    def get_reward(m1, m2):
        if m1 == m2:
            return 0
        if (m1, m2) in {("R","S"),("S","P"),("P","R")}:
            return 1
        return -1

    def last_two_str(moves):
        if len(moves) < 2:
            return (""*(2-len(moves))) + "".join(moves)
        return "".join(moves[-2:])

    def init_q(state, act):
        if (state, act) not in q_table:
            q_table[(state, act)] = random.uniform(0, 0.5)

    def best_act(state):
        for a in "RPS":
            init_q(state, a)
        return max("RPS", key=lambda x: q_table[(state, x)])

    # Detect a new match
    if prev_play == "" and not opponent_history:
        q_table.clear()
        last_state[0] = ("", "")
        last_action[0] = "R"

    # Track opponent's move
    if prev_play in "RPS":
        opponent_history.append(prev_play)

    # Prepare our "my_moves" array in q_table for storing our plays
    if "my_moves" not in q_table:
        q_table["my_moves"] = []
    if not q_table["my_moves"] or q_table["my_moves"][-1] != last_action[0]:
        q_table["my_moves"].append(last_action[0])

    # Build current state: (opp_last2, our_last2)
    opp_s = last_two_str(opponent_history)
    my_s = last_two_str(q_table["my_moves"])
    cur_state = (opp_s, my_s)

    # Epsilon-greedy action
    for a in "RPS":
        init_q(cur_state, a)
    import random
    if random.random() < epsilon[0]:
        action = random.choice("RPS")
    else:
        action = best_act(cur_state)

    # Q-update for the previous round
    # old_state/old_action -> reward -> new_state
    if prev_play in "RPS":
        old_state = last_state[0]
        old_action = last_action[0]
        r = get_reward(old_action, prev_play)

        # Initialize Q for old pair in case it's missing
        init_q(old_state, old_action)

        # Our hypothetical new state
        new_my_moves = q_table["my_moves"] + [action]
        new_state = (opp_s, last_two_str(new_my_moves))

        for a in "RPS":
            init_q(new_state, a)

        old_q = q_table[(old_state, old_action)]
        best_future = q_table[(new_state, best_act(new_state))]
        q_table[(old_state, old_action)] = old_q + alpha*(r + gamma*best_future - old_q)

    # Update "last_state" and "last_action"
    last_state[0] = cur_state
    last_action[0] = action

    # Record our new action
    q_table["my_moves"].append(action)

    return action