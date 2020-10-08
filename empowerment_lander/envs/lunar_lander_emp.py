import sys, math
import numpy as np

from copy import deepcopy

# import sys
# sys.path.insert(0, "/usr/local/lib/python3.7/site-packages/Box2D")

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import csv
import gym
import copy
from gym import spaces
from gym.utils import seeding, EzPickle
from scipy.spatial import ConvexHull
import time

from utils.env_utils import *

from pickle import dumps, loads

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

MAX_NUM_STEPS = 1000

OBS_DIM = 9
ACT_DIM = 6

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [
    (-14, +17), (-17, 0), (-17, -10),
    (+17, -10), (+17, 0), (+14, +17)
]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

NUM_CONCAT = 20


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLanderEmpowerment(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = False

    def __init__(self, empowerment, ac_continuous=True, pilot_policy=None, pilot_is_human=False, log_file=None, **extras):
        EzPickle.__init__(self, empowerment, ac_continuous)
        self.seed()
        self.viewer = None
        self.num_concat = NUM_CONCAT
        self.act_dim = ACT_DIM
        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []
        self.pilot_is_human = pilot_is_human
        self.prev_reward = None
        self.log_file = log_file

        self.continuous = ac_continuous
        self.pilot_policy = pilot_policy
        self.copilot = False
        if self.pilot_policy is not None:
            self.copilot = True
            self.past_pilot_actions = np.zeros((NUM_CONCAT * ACT_DIM))

        self.fake_step = False

        # useful range is -1 .. +1, but spikes can be higher
        obs_box = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

        if self.copilot:
            # consider action from pilot as well
            self.observation_space = spaces.Box(np.concatenate((obs_box.low, np.zeros(NUM_CONCAT * ACT_DIM))),
                                         np.concatenate((obs_box.high, np.ones(NUM_CONCAT * ACT_DIM))))
        else:
            self.observation_space = obs_box

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(ACT_DIM)

        self.empowerment_coeff = empowerment
        self.curr_step = 0
        self.trajectory = None
        self.actions = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.curr_step = 0
        self.actions = []
        self.trajectory = []
        self.num_steps_at_site = 0
        self.prev_at_site = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        helipad_chunk = np.random.choice(range(1, CHUNKS-1)) # randomize helipad x coordinate

        self.helipad_x1 = chunk_x[helipad_chunk-1]
        self.helipad_x2 = chunk_x[helipad_chunk+1]
        self.helipad_y = H / 4
        height[helipad_chunk - 2] = self.helipad_y
        height[helipad_chunk - 1] = self.helipad_y
        height[helipad_chunk + 0] = self.helipad_y
        height[helipad_chunk + 1] = self.helipad_y
        height[helipad_chunk + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(1)[0] #self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        action = disc_to_cont(action)

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if action[0] > 0.0:
        #if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            assert m_power >= 0.5 and m_power <= 1.0

            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[
                1]  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if not self.fake_step:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)  # particles are just a decoration, 3.5 is here to make particle speed adequate ###
                p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos, True)

        s_power = 0.0
        if np.abs(action[1]) > 0.5:
        #if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            assert s_power >= 0.5 and s_power <= 1.0

            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)

            if not self.fake_step:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True) ###
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        helipad_x = (self.helipad_x1 + self.helipad_x2) / 2
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (helipad_x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        ]
        assert len(state) == OBS_DIM

        self.curr_step += 1
        if not self.fake_step:
            self.trajectory.append(state)
            self.actions.append(action)

        reward = 0
        dx = (pos.x - helipad_x) / (VIEWPORT_W/SCALE/2)

        # TODO: add option to remove shaping if necessary
        shaping = \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]# And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if not self.copilot:
            shaping = shaping -100 * np.sqrt(dx * dx + state[1] * state[1])  #only if we're the pilot do we know where the goal is
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power * 0.03

        ## Empowerment
        if self.empowerment_coeff > 0 and not self.fake_step:# and not(self.legs[0].ground_contact and self.legs[1].ground_contact):
            height_scale = 1#(pos.y - self.helipad_y) / (VIEWPORT_H / SCALE)
            obs_dim = OBS_DIM
            emp = self.compute_empowerment(state, obs_dim)
            reward += self.empowerment_coeff * height_scale * emp

        timeout = self.curr_step >= MAX_NUM_STEPS
        at_site = pos.x >= self.helipad_x1 and pos.x <= self.helipad_x2 and self.legs[0].ground_contact and self.legs[1].ground_contact

        if at_site:
            self.num_steps_at_site += 1
        else:
            self.num_steps_at_site = 0


        done = self.game_over or abs(state[0]) >= 1.0 or timeout or not self.lander.awake or self.num_steps_at_site > 3

        info = {}
        if done and not self.fake_step:
            if self.game_over or abs(state[0]) >= 1.0 or timeout:
                reward = -100
            elif at_site:
                reward = +100
            print(reward)
            info['trajectory'] = self.trajectory
            info['actions'] = self.actions
            trajectory = np.asarray(self.trajectory)
            with open(self.log_file + '_dist.csv', 'a', newline='') as logfile:
                writer = csv.writer(logfile, delimiter=',')
                dist_to_goal = np.sqrt(trajectory[:,0] ** 2 + trajectory[:,1] ** 2)
                writer.writerow([reward] + list(dist_to_goal))

            with open(self.log_file + '_speed.csv', 'a', newline='') as logfile:
                writer = csv.writer(logfile, delimiter=',')
                speed = np.sqrt(trajectory[:,2] ** 2 + trajectory[:,3] ** 2)
                writer.writerow([reward] + list(speed))

            with open(self.log_file + '_angle.csv', 'a', newline='') as logfile:
                writer = csv.writer(logfile, delimiter=',')
                angle = trajectory[:,4]
                writer.writerow([reward] + list(angle))

        state = np.array(state, dtype=np.float32)

        if self.copilot and not self.fake_step:
            if self.pilot_is_human:
                pilot_action = onehot_encode(self.pilot_policy(state[None,:]))
            else:
                pilot_action = onehot_encode(self.pilot_policy.step(state[None, :]))
            self.past_pilot_actions[ACT_DIM * 1:] = self.past_pilot_actions[:-1 * ACT_DIM]
            self.past_pilot_actions[:ACT_DIM] = pilot_action
            state = np.concatenate((state, self.past_pilot_actions))

        return state, reward, done, info

    def render(self, mode='human', close=False):
        if close:
            self.close()

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def compute_empowerment(self, state, state_dim, horizon=5, n_traj=10):
        """
        Estimate empowerment using a convex hull approximation.
        :param env: Environment
        :param state: State from which to compute empowerment
        :param state_dim: dimension of state space
        :param horizon: time horizon
        :param n_traj: number of action trajectories (final states)
        :return: Volume estimation of empowerment
        """
        # array for the final state
        X = np.zeros((n_traj, state_dim))
        self.fake_step = True

        lander_pos = (self.lander.position.x, self.lander.position.y)
        lander_ang_vel = self.lander.angularVelocity
        lander_lin_vel = (self.lander.linearVelocity.x, self.lander.linearVelocity.y)
        lander_angle = self.lander.angle

        leg_pos = []
        leg_joint = []
        for leg in self.legs:
            leg_pos.append((leg.position.x, leg.position.y))
            leg_joint.append(leg.joint)

        particle_pos = []
        for particle in self.particles:
            particle_pos.append((particle.position.x, particle.position.y))

        orig_curr_step = deepcopy(self.curr_step)
        orig_shaping = deepcopy(self.prev_shaping)
        orig_game_over = deepcopy(self.game_over)
        num_steps_at_site = deepcopy(self.num_steps_at_site)

        for n in range(n_traj):
            # generate an action sequence.
            for _ in range(horizon):
                # step in environment using random action a
                a = self.action_space.sample()
                x, rew, done, info = self.step(a)
                if done:
                    break
            # store the final state
            X[n, :] = x

            for pos, joint, leg in zip(leg_pos,leg_joint,self.legs):
                leg.position = pos
                leg.joint = joint

            for pos, particle in zip(particle_pos, self.particles):
                particle.position = pos

            self.lander.position = lander_pos
            self.lander.angularVelocity = lander_ang_vel
            self.lander.linearVelocity = lander_lin_vel
            self.lander.angle = lander_angle
            self.curr_step = deepcopy(orig_curr_step)
            self.prev_shaping = deepcopy(orig_shaping)
            self.game_over = deepcopy(orig_game_over)
            self.num_steps_at_site = deepcopy(num_steps_at_site)
            # compute the convex hull of the final state
        # e.g., by scipy.spatial.ConvexHull
        try:
            fs = X[:,:2]
            est_emp = np.var(fs)
            #ch = ConvexHull(fs)

            # take volume
            #ch_volume = ch.volume
        except Exception as e:
            est_emp = 0
        self.fake_step = False
        return est_emp

