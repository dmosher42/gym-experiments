import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class ThrustboxEnvZ(MujocoEnv, utils.EzPickle):
    """
    ## Description

    The ThrustBox0 is a 3D robot consisting of one torso (free rotational body) with 24 thrusters
    attached to it allowing for 3D movement.

    The goal is to move the Thrustbox to a random point.

    ## Action Space
    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    v0 MVP Testing
    +z motion (all +z thrustsers tied together)
    -z motion (all -z thrustsers tied together)

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Thruster 1 in the +Z direction                                    | 0           | 1           | motor+Z+x+y                      | gear  | 				|
    | 1   | Thruster 2 in the +Z direction                                    | 0           | 1           | motor+Z-x+y                      | gear  | 				|
    | 2   | Thruster 3 in the +Z direction                                    | 0           | 1           | motor+Z-x-y                      | gear  | 				|
    | 3   | Thruster 4 in the +Z direction                                    | 0           | 1           | motor+Z+x-y                      | gear  | 				|
    | 4   | Thruster 1 in the -Z direction                                    | 0           | 1           | motor-Z+x+y                      | gear  | 				|
    | 5   | Thruster 2 in the -Z direction                                    | 0           | 1           | motor-Z-x+y                      | gear  | 				|
    | 6   | Thruster 3 in the -Z direction                                    | 0           | 1           | motor-Z-x-y                      | gear  | 				|
    | 7   | Thruster 4 in the -Z direction                                    | 0           | 1           | motor-Z+x-y                      | gear  | 				|
    | 8   | Thruster 1 in the +Y direction                                    | 0           | 1           | motor+Y+x+z                      | gear  | 				|
    | 9   | Thruster 2 in the +Y direction                                    | 0           | 1           | motor+Y+x-z                      | gear  | 				|
    | 10  | Thruster 3 in the +Y direction                                    | 0           | 1           | motor+Y-x-z                      | gear  | 				|
    | 11  | Thruster 4 in the +Y direction                                    | 0           | 1           | motor+Y-x+z                      | gear  | 				|
	| 12  | Thruster 1 in the -Y direction                                    | 0           | 1           | motor-Y+x+z                      | gear  | 				|
    | 13  | Thruster 2 in the -Y direction                                    | 0           | 1           | motor-Y+x-z                      | gear  | 				|
    | 14  | Thruster 3 in the -Y direction                                    | 0           | 1           | motor-Y-x-z                      | gear  | 				|
    | 15  | Thruster 4 in the -Y direction                                    | 0           | 1           | motor-Y-x+z                      | gear  | 				|
	| 16  | Thruster 1 in the +X direction                                    | 0           | 1           | motor+X+y+z                      | gear  | 				|
    | 17  | Thruster 2 in the +X direction                                    | 0           | 1           | motor+X-y+z                      | gear  | 				|
    | 18  | Thruster 3 in the +X direction                                    | 0           | 1           | motor+X-y-z                      | gear  | 				|
    | 19  | Thruster 4 in the +X direction                                    | 0           | 1           | motor+X+y-z                      | gear  | 				|
	| 20  | Thruster 1 in the -X direction                                    | 0           | 1           | motor-X+y+z                      | gear  | 				|
    | 21  | Thruster 2 in the -X direction                                    | 0           | 1           | motor-X-y+z                      | gear  | 				|
    | 22  | Thruster 3 in the -X direction                                    | 0           | 1           | motor-X-y-z                      | gear  | 				|
    | 23  | Thruster 4 in the -X direction                                    | 0           | 1           | motor-X+y-z                      | gear  | 				|

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
    2   +/- z coords of client
    3   +/- Linear z velocity


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
    | 12  | x-coordinate of the target                                   | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | 13  | y-coordinate of the target                                   | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | 14  | z-coordinate of the target                                   | -Inf   | Inf    | torso                                  | free  | position (m)             |


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
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "C:\More Projects\Vissidus\gym-examples\gym_examples\envs\/assets\/thrustbox_target_z_only_v0.xml",
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, a):
        vec = self.vec_boxtarget()
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()

        reward = reward_dist + reward_ctrl #+ reward_deltav

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        # TODO: add done
        done = False

        # TODO: add info
        info = False

        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            info,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-2, high=2, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
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

