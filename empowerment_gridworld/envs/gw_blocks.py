import numpy as np
from gym.utils import seeding
from gym import error, spaces, utils
import gym
from enum import IntEnum
from gym.envs.toy_text import discrete
from six import StringIO
import sys
from contextlib import closing


class GridWorldEnv(discrete.DiscreteEnv):

    class Actions(IntEnum):
        left = 0
        down = 1
        right = 2
        up = 3
        stay = 4

    def __init__(self, seed, grid_size, p, human_pos, boxes_pos, human_goal):
        """
        Gridworld environment with blocks.

        :param size: gridworld dimensions are (size, size)
        :param p:
        :param num_blocks:
        """

        self.num_boxes = int(len(boxes_pos)/2)
        assert self.num_boxes > 0, "Cannot have 0 Boxes"

        self.actions = GridWorldEnv.Actions
        self.action_dim = 2 # one for action, one for box number
        nA = len(self.actions) * self.num_boxes
        self.nA = nA

        self.state_dim = 2 + self.num_boxes*2  #my coordinates, goal coordinates, and coordinates of boxes
        nS = grid_size ** self.state_dim
        self.grid_size = grid_size

        self.p = p

        self.seed(seed)

        isd = np.zeros(nS)

        self.cur_pos = human_pos
        self.boxes_pos = boxes_pos
        isd[self.to_s(np.concatenate((self.cur_pos, self.boxes_pos)))] = 1.0

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        #populate P
        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                if self.p == 1.0:
                    s_next = self.to_s(self.inc_boxes(self.from_s(s),self.from_a(a)))
                    li.append((1.0, s_next, 0, False)) # prob, next_s, rew, done
                else:
                    raise NotImplementedError

        self.human_goal = human_goal

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def inc_boxes(self, state_vec, a):
        """
        In
        :param state_vec:
        :param a:
        :return:
        """
        row, col = state_vec[0], state_vec[1]

        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]

        for cur_box in range(self.num_boxes):
            box, ac = a[1], a[0]
            if box is not cur_box:
                continue
            else:
                b_col = b_cols[box]
                b_row = b_rows[box]

                other_cols = np.copy(b_cols)
                other_cols[box] = col #replace with human pos
                other_rows = np.copy(b_rows)
                other_rows[box] = row

                if ac == self.actions.left:
                    b_col = self.inc_(b_col, b_row, other_cols, other_rows, -1)
                elif ac == self.actions.down:
                    b_row = self.inc_(b_row, b_col, other_rows, other_cols, 1)
                elif ac == self.actions.right:
                    b_col = self.inc_(b_col, b_row, other_cols, other_rows, 1)
                elif ac == self.actions.up:
                    b_row = self.inc_(b_row, b_col, other_rows, other_cols, -1)
                elif ac == self.actions.stay:
                    pass

                b_cols[box] = b_col
                b_rows[box] = b_row

        return [row, col] + [*sum(zip(b_rows, b_cols), ())]

    def inc_(self, pos_move, pos_other, other_pos_move, other_pos_other, delta):
        target_block = False  # if target pos has a block or human, can't move block there
        for i in range(self.num_boxes):
            if (pos_move + delta, pos_other) == (other_pos_move[i], other_pos_other[i]):
                target_block = True
        if not target_block:
            pos_move = min(max(pos_move + delta, 0), self.grid_size - 1)
        return pos_move

    def to_s(self, positions):
        return np.sum([pos * (self.grid_size ** i) for i, pos in enumerate(positions)])

    def from_s(self, s):
        state_vec = []
        for i in range(self.state_dim):
            state_vec.append(s % self.grid_size)
            s //= self.grid_size
        return state_vec

    def to_a(self, action):
        return action[0] + action[1] * len(self.actions)

    def from_a(self, a):
        action_vec = []
        action_vec.append(a % len(self.actions))
        action_vec.append(a // len(self.actions))
        return action_vec

    def set_state(self, s):
        self.s = s

    def step_human(self, s, human_goal=None):
        if human_goal == None:
            human_goal = self.human_goal

        state_vec = self.from_s(s)

        best_row = None
        best_col = None
        dist = np.inf
        best_ac = None
        for ac in range(5):
            row, col = state_vec[0], state_vec[1]  # current human position
            b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
            b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]  # boxes cols

            if ac == self.actions.left:
                col = self.inc_(col, row, b_cols, b_rows, -1)
            elif ac == self.actions.down:
                row = self.inc_(row, col, b_rows, b_cols, 1)
            elif ac == self.actions.right:
                col = self.inc_(col, row, b_cols, b_rows, 1)
            elif ac == self.actions.up:
                row = self.inc_(row, col, b_rows, b_cols, -1)
            elif ac == self.actions.stay:
                pass

            # find the action that brings the human closest to its goal
            cur_dist = np.linalg.norm(np.asarray([row, col]) - human_goal)

            if cur_dist < dist:
                dist = cur_dist
                best_row = row
                best_col = col
                best_ac = ac

        new_state = [best_row, best_col] + [*sum(zip(b_rows, b_cols), ())]

        done = np.array_equal([best_row, best_col], human_goal)

        return self.to_s(new_state), done, best_ac

    def human_dist_to_goal(self, s, goal_states):
        # return distance to each goal in goal_states
        state_vec = self.from_s(s)
        row, col = state_vec[0], state_vec[1]  # current human position
        dist_to_goal = {}
        for goal in goal_states:
            dist_to_goal[goal] = np.linalg.norm(np.asarray([row, col]) - np.asarray(goal))

        return dist_to_goal

    def infer_a(self, s, human_goal):
        #compute action to help human move towards their goal
        state_vec = self.from_s(s)
        h_row, h_col = state_vec[0], state_vec[1]  # current human position
        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]  # boxes rows
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]  # boxes cols

        dist = np.inf

        for human_ac in range(5):
            row = h_row
            col = h_col

            if human_ac == self.actions.left:
                col = max(col - 1, 0)
            elif human_ac == self.actions.down:
                row = min(row + 1, self.grid_size-1)
            elif human_ac == self.actions.right:
                col = min(col + 1, self.grid_size-1)
            elif human_ac == self.actions.up:
                row = max(row - 1, 0)
            elif human_ac == self.actions.stay:
                pass

            # find the action that brings the human closest to its goal
            cur_dist = np.linalg.norm(np.asarray([row, col]) - human_goal)

            if cur_dist < dist:
                dist = cur_dist
                best_row = row
                best_col = col

        action = []

        #find box we interfere with
        for i, (b_row, b_col) in enumerate(zip(b_rows, b_cols)):
            if best_row == b_row and best_col == b_col:
                box = i #first index of action is the box we're moving

                other_cols = np.copy(b_cols)
                other_cols[box] = h_col
                other_rows = np.copy(b_rows)
                other_rows[box] = h_row

                for box_a in self.actions:
                    if box_a == self.actions.left:
                        b_col_new = self.inc_(b_col, b_row, other_cols, other_rows, -1)
                        if b_col_new != b_col: #can move this block
                            action.append(box_a)
                            action.append(box)
                            break
                    elif box_a == self.actions.down:
                        b_row_new = self.inc_(b_row, b_col, other_rows, other_cols, 1)
                        if b_row_new != b_row:
                            action.append(box_a)
                            action.append(box)
                            break
                    elif box_a == self.actions.right:
                        b_col_new = self.inc_(b_col, b_row, other_cols, other_rows, 1)
                        if b_col_new != b_col: #can move this block
                            action.append(box_a)
                            action.append(box)
                            break

                    elif box_a == self.actions.up:
                        b_row_new = self.inc_(b_row, b_col, other_rows, other_cols, -1)
                        if b_row_new != b_row:
                            action.append(box_a)
                            action.append(box)
                            break

                    elif box_a == self.actions.stay:
                        pass
        if len(action) == 0:
            return 4
        return self.to_a(action)

    def render(self,  filename=None, mode='human'):
        if filename is None:
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            colorize = True
        else:
            outfile = open(filename, "a")
            colorize = False

        state_vec = self.from_s(self.s)
        row, col = state_vec[0], state_vec[1]
        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]
        goal_row, goal_col = self.human_goal[0], self.human_goal[1]

        desc = [["0" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        desc[row][col] = "1"
        if colorize:
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        desc[goal_row][goal_col] = "3"
        if colorize:
            desc[goal_row][goal_col] = utils.colorize(desc[goal_row][goal_col], "green", highlight=True)

        for box_row, box_col in zip(b_rows, b_cols):
            desc[box_row][box_col] = "2"
            if colorize:
                desc[box_row][box_col] = utils.colorize(desc[box_row][box_col], "blue", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up","Stay"][self.lastaction % len(self.actions)]))
        else:
            outfile.write("\n")

        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if filename is not None:
            outfile.close()
