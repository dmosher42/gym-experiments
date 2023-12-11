__credits__ = ["Dee Mosher"]

import random

import numpy as np
import logging
import gym_examples.util

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from pathlib import Path
import gym_examples.thrustboxbuilder

# Todo: write setup.py + other setup things


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
}
logging.basicConfig()

# create logger from Logging Cookbook
module_logger = logging.getLogger('z_axis_application').getChild("SimpleBoxEnv")

class SimpleThrustboxEnvZ(MujocoEnv, utils.EzPickle):
    """
    ## Description

    The SimpleThrustBox0 is a 3D robot consisting of one torso (free rotational body) and 3D movement.

    The goal is to move the Thrustbox from a random point to the origin.

    ## Action Space
    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    v0 MVP Testing

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Thruster in the +/-X direction                                    | -1          | 1           | motorX                           | gear  | 				|
    | 1   | Thruster in the +/-Y direction                                    | -1          | 1           | motorY                           | gear  | 				|
    | 2   | Thruster in the +/-Z direction                                    | -1          | 1           | motorZ                           | gear  | 				|
    | 3   | Rotate around the X axis                                          | -1          | 1           | motorX                           | gear  | 				|
    | 4   | Rotate around the Y axis                                          | -1          | 1           | motorY                           | gear  | 				|
    | 5   | Rotate around the Z axis                                          | -1          | 1           | motorZ                           | gear  | 				|

   ## Observation Space

    Observations consist of positional values of different body parts of the ant,
    followed by the velocities of those individual parts (their derivatives) with all
    the positions ordered before all the velocities.

    By default, observations do not include the x- and y-coordinates of the ant's torso. These may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 29 dimensions where the first two dimensions
    represent the x- and y- coordinates of the ant's torso.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
    of the torso will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.

    However, by default, an observation is a `ndarray` with shape `(27,)`
    where the elements correspond to the following:

    v0 testing:
    1   +/- z coords of robot; 3d, -inf, inf
    2   +/- Linear z velocity
    3   +/- z coords of client



    in `v0` or earlier:
    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Unit                     |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | x-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | 1   | y-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | 2   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | 3   | x-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 4   | y-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 5   | z-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 6   | x-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
    | 7   | y-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
    | 8   | z-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
    | 9   | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
    | 10  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
    | 11  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |


   ## Rewards
    The reward consists of three parts:
    - *healthy_reward*:
    - *forward_reward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the time
    between actions and is dependent on the `frame_skip` parameter (default is 5),
    where the frametime is 0.01 - making the default *dt = 5 * 0.01 = 0.05*.
    This reward would be positive if the ant moves forward (in positive x direction).
    - *ctrl_cost*: A negative reward for penalising the ant if it takes actions
    that are too large. It is measured as *`ctrl_cost_weight` * sum(action<sup>2</sup>)*
    where *`ctr_cost_weight`* is a parameter set for the control and has a default value of 0.5.
    - *contact_cost*: A negative reward for penalising the ant if the external contact
    force is too large. It is calculated *`contact_cost_weight` * sum(clip(external contact
    force to `contact_force_range`)<sup>2</sup>)*.

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.

    But if `use_contact_forces=True` or version < `v4`
    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.

    In either case `info` will also contain the individual reward terms.

    ## Starting State
    All observations start in state
    (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a uniform noise in the range
    of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional values and standard normal noise
    with mean 0 and standard deviation `reset_noise_scale` added to the velocity values for
    stochasticity. Note that the initial z coordinate is intentionally selected
    to be slightly high, thereby indicating a standing up ant. The initial orientation
    is designed to make it face forward as well.

    ## Episode End
    The ant is said to be unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite
    2. The z-coordinate of the torso is **not** in the closed interval given by `healthy_z_range` (defaults to [0.2, 1.0])

    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode ends when any of the following happens:

    1. Truncation: The episode duration reaches a 1000 timesteps
    2. Termination: The ant is unhealthy

    If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

    ## Arguments

    No additional arguments are currently supported in v2 and lower.

    ```python
    import gymnasium as gym
    env = gym.make('Ant-v2')
    ```

    v3 and v4 take `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('Ant-v4', ctrl_cost_weight=0.1, ...)
    ```

    | Parameter               | Type       | Default      |Description                    |
    |-------------------------|------------|--------------|-------------------------------|


    ## Version History

    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50, # TODO: lower this
    }
    def __init__(self, usebuilder=False, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64)
        if usebuilder:
            model_path=str(Path.cwd() / "thrustboxbuilder_temp.xml")
        else:
            model_path="C:\More Projects\Vissidus\gym-examples\gym_examples\envs\/assets\/simplethrustbox_target_z_v0.xml"



        MujocoEnv.__init__(
            self,
            model_path,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        logging.basicConfig()
        self.logger = logging.getLogger('Simplebox_z_v0')
        # self.logger.debug("Started Logging Env")
        # self.logger.addFilter(gym_examples.util.DuplicateFilter())  # add the filter to it
        # set camera to render from
        self.camera_name="camera1"

    def step(self, a):
        truncated = False
        terminated = False
        outbounds = False
        reward_near = 0
        vec = self.vec_boxtarget()
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()

        reward = reward_near + reward_dist + reward_ctrl #+ reward_deltav+ reward_deltatheta

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        # Out of bounds failure
        if -reward_dist>10:
            self.logger.debug("Out of Bounds")
            reward = reward - 1000
            # print ("Out of bounds")
            terminated = True
            outbounds = True

        # Near Target success
        # if -reward_dist<.1:
        #     print("Close enough")
        #     reward = reward + 1000
        #     terminated = True

        # TODO: add info
        info = False

        ob = self._get_obs()
        return (
            ob,
            reward,
            terminated,
            truncated,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, oob=outbounds),
        )

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-2, high=2, size=self.model.nq)
            + self.init_qpos
        )
        # print(qpos)
        # while True:
        self.goal = self.np_random.uniform(low=-3, high=3, size=3)

        qpos[7:10] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[7:10] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def vec_boxtarget(self):
        return self.get_body_com("box") - self.get_body_com("target")

    def _get_obs(self):
        # print(self.data.qpos)
        return np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                self.vec_boxtarget()
            ]
        )

