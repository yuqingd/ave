import numpy as np


class EmpowermentCountingPolicy:

    def __init__(self, env, horizon=10, num_traj=1000, estimate_emp=False, account_for_human=False, goal_oriented=False, proxy=False):

        self.env = env
        self.act_dim = env.action_space.n
        self.horizon = horizon
        self.num_traj = num_traj
        self.estimate_emp = estimate_emp
        self.account_for_human = account_for_human
        self.goal_oriented = goal_oriented
        self.proxy = proxy

    def compute_n_step_empowerment(self, s):
        """
        Compute empowerment from being in state s
        :param s:
        :return:
        """

        if self.estimate_emp: #randomly sample from possible action trajectories
            actions = np.random.choice(self.act_dim, size=(self.num_traj, self.horizon))
        else:
            actions = np.array(np.meshgrid(*np.repeat([np.arange(self.act_dim)], self.horizon, axis=0)))
            actions = actions.reshape(self.horizon, -1).T

        seen_states = set()

        if self.goal_oriented:
            human_seen_states = self.compute_human_empowerment(s)
            seen_states.update(human_seen_states)
            seen_states.add(-1)
        else:
            for a_seq in actions:
                self.env.set_state(s)

                for a in a_seq:
                    s_next, _, _, _ = self.env.step(a)
                seen_states.add(s_next)
                if self.account_for_human:
                    human_seen_states = self.compute_human_empowerment(s_next)
                    seen_states.update(human_seen_states)


        if self.proxy:
            return np.log(np.var(list(seen_states)))
        else:
            return np.log(len(seen_states))

    def compute_human_empowerment(self, s, h_num_traj=1, h_horizon=1):
        state_vec = self.env.from_s(s)
        seen_states = set()

        h_row, h_col = state_vec[0], state_vec[1]

        other_rows = [state_vec[i] for i in range(2, self.env.state_dim - 1, 2)]
        other_cols = [state_vec[i] for i in range(3, self.env.state_dim, 2)]
        if not self.goal_oriented:
            actions_h = np.random.choice(self.act_dim, size=(h_num_traj, h_horizon))

            for ac_seq in actions_h:
                for ac_h in ac_seq:
                    if ac_h == self.env.actions.left:
                        h_col = self.env.inc_(h_col, h_row, other_cols, other_rows, -1)
                    elif ac_h == self.env.actions.down:
                        h_row = self.env.inc_(h_row, h_col, other_rows, other_cols, 1)
                    elif ac_h == self.env.actions.right:
                        h_col = self.env.inc_(h_col, h_row, other_cols, other_rows, 1)
                    elif ac_h == self.env.actions.up:
                        h_row = self.env.inc_(h_row, h_col, other_rows, other_cols, -1)
                    elif ac_h == self.env.actions.stay:
                        pass

                new_h_s = [h_row, h_col] + [*sum(zip(other_rows, other_cols), ())]
                new_h_s = self.env.to_s(new_h_s)
                if new_h_s != s:
                    seen_states.add(new_h_s)

        else:
            new_h_s, done, _ = self.env.step_human(s)
            if new_h_s != s:
                seen_states.add(new_h_s)

        return seen_states


    def next_action(self, s):
        empowerment = np.zeros(self.act_dim)
        for a in range(self.act_dim):
            self.env.set_state(s)
            s_next, _, _, _ = self.env.step(a)
            empowerment[a] = self.compute_n_step_empowerment(s_next)

        return np.argmax(empowerment)

    def compute_empowerment_matrix(self):
        emp = np.zeros((self.env.nrow, self.env.ncol))
        for row in range(self.env.nrow):
            for col in range(self.env.ncol):
                emp[row][col] = self.compute_n_step_empowerment(self.env.to_s(row, col))
        return emp
