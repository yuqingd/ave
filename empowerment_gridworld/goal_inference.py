import numpy as np


class GoalInferencePolicy:

    def __init__(self, env, goal_set, alpha=0.5, plan_expectation=True):
        self.env = env
        self.alpha = alpha
        self.goal_set = goal_set #set of potential human goals

        self.goal_probs = {} #for each goal, what is the probability, init all equal
        self.num_goals = len(self.goal_set)
        for goal in self.goal_set:
            self.goal_probs[goal] = 1/self.num_goals #assign equal prob to all at first
        self.plan_expectation = plan_expectation

    def next_action(self, s, s_prev):
        action = -1 #don't move anything

        prev_dist_to_goal = self.env.human_dist_to_goal(s_prev, self.goal_set)
        cur_dist_to_goal = self.env.human_dist_to_goal(s, self.goal_set)

        # for each goal, check if human moved closer or farther
        prob_adj = 1/self.num_goals * self.alpha
        # if closer, increase prob, if farther, decrease prob

        for goal in self.goal_set:
            delta = cur_dist_to_goal[goal] - prev_dist_to_goal[goal]
            if delta > 0: #move away from this goal
                if prob_adj > self.goal_probs[goal]:
                    self.goal_probs[goal] = 0
                else:
                    self.goal_probs[goal] -= prob_adj
            elif delta < 0:
                self.goal_probs[goal] += prob_adj
            else:
                if cur_dist_to_goal[goal] == 0:
                    self.goal_probs[goal] = 0
        # normalize probs
        total_prob = np.sum(list(self.goal_probs.values()))
        self.goal_probs = {k : v / total_prob for k,v in self.goal_probs.items()}

        if self.plan_expectation:
            # find action most likely to help with goals
            actions = {}
            for ac in range(self.env.nA):
                actions[ac] = 0

            for goal in self.goal_set:

                best_action = self.env.infer_a(s, goal)
                actions[best_action] += self.goal_probs[goal]

            mx = max(actions.values())
            actions = [k for k, v in actions.items() if v == mx]
            action_idx = np.random.choice(len(actions), 1)[0]
            action = actions[action_idx]

        else:
            # from probs, sample which goal we think human is going to
            idx = np.arange(len(self.goal_probs.keys()))
            sample_goal_idx = np.random.choice(idx, 1, p=np.asarray(list(self.goal_probs.values())))
            sample_goal = list(self.goal_probs.keys())[sample_goal_idx[0]]

            # find human action that brings human closest to that goal (assume greedy)
            # if block is in the way of that action, then move block out of the way
            action = self.env.infer_a(s, sample_goal)

        return action