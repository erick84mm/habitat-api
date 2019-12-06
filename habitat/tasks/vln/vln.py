#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import attr
import json
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
    heading_to_rotation,
)
from habitat.utils.visualizations import fog_of_war, maps

cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 1250


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None and
        episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class InstructionData:
    r"""Base class for Instruction processing.
    """
    instruction: str
    tokens: Optional[List[int]] = None


@attr.s(auto_attribs=True, kw_only=True)
class ViewpointData:
    r"""Base class for a viewpoint specification.
    """
    image_id: str = attr.ib(default=None, validator=not_none_validator)
    view_point: AgentState = None
    radius: Optional[float] = None



@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, navigation instruction and optional shortest
    paths. An episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset. From path_id in R2R
        scene_id: id of scene in scene dataset. From scan in R2R
        start_position: numpy ndarray containing 3 entries for (x, y, z).
            From viewpoint_to_xyz in R2R.
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor.
            From heading_to_rotation in R2R.
        goals: list of viewpoints in R2R path.
        instruction: the instruction in R2R.
        scan: The name of the scan in R2R.
        curr_viewpoint: In the case we are running the task in discrete mode
        this variable holds the image_id of the place the agent is at.
    """
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: List[ViewpointData] = attr.ib(
        default=None, validator=not_none_validator
    )
    scan: str = None
    curr_viewpoint: Optional[str] = None



@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return self._quat_to_xy_heading(rotation_world_agent.inverse())


@registry.register_sensor
class AdjacentViewpointSensor(Sensor):
    r"""Sensor for observing the adjacent viewpoints near the current
    position of the agent. Created for the discrete VLNTask.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        connectivity_path = getattr(config, "CONNECTIVITY_PATH", "")
        self._connectivity = self._load_connectivity(connectivity_path)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "adjacentViewpoints"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NULL  # Missing sensor type

    def _load_connectivity(self, path):
        data = {}
        if path:
            with open(path) as f:
                data = json.load(f)
        return data

    def _quat_to_xy_heading_vector(self, quat):
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(quat, direction_vector)
        return heading_vector

    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self,v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'"""
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _is_navigable(self, target_pos):
        '''
        For a viewpoint to be accessible it has to be within the
        horizontal field of View HFOV.

        This function returns True if the target position is
        accessible from the curr_viewpoint given the previous
        condition.
        '''
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        heading_vector = self._quat_to_xy_heading_vector(
                            rotation_world_agent.inverse()
        )
        target_vector = np.array(target_pos) - np.array(agent_state.position)

        angle = self._angle_between(
            heading_vector,
            target_vector
        )
        rot = heading_to_rotation(angle)
        opposite_angle = 2 * np.pi - angle
        target_angle = self._sim.config.RGB_SENSOR.HFOV * 2 * np.pi / 360 / 2

        if angle <= target_angle or opposite_angle <= target_angle:
                return True
        return False

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        observations = []
        if kwargs and 'scan' in kwargs:
            scan = kwargs["scan"]
            curr_viewpoint = kwargs["curr_viewpoint"]
            scan_inf = self._connectivity[scan]
            viewpoint_inf = scan_inf["viewpoints"][curr_viewpoint]
            observations.append(
                    {
                        "image_id": curr_viewpoint,
                        "start_position":
                            viewpoint_inf["start_position"],
                        "start_rotation":
                            viewpoint_inf["start_rotation"]
                    }
            )
            for i in range(len(viewpoint_inf["visible"])):
                if viewpoint_inf["included"] \
                and viewpoint_inf["unobstructed"][i] \
                and viewpoint_inf["visible"][i]:
                    adjacent_viewpoint_name = scan_inf["idxtoid"][str(i)]
                    adjacent_viewpoint = \
                        scan_inf["viewpoints"][adjacent_viewpoint_name]
                    if adjacent_viewpoint["included"]:
                        observations.append(
                            {
                                "image_id": adjacent_viewpoint_name,
                                "start_position":
                                    adjacent_viewpoint["start_position"],
                                "start_rotation":
                                    adjacent_viewpoint["start_rotation"]
                            }
                        )
        return observations

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        abjacent_viewpoints = self._get_observation_space(
            scan=episode.scan,
            curr_viewpoint=episode.curr_viewpoint
            )
        navigable_viewpoints = []
        for viewpoint in abjacent_viewpoints:
            target_pos = viewpoint["start_position"]
            if self._is_navigable(target_pos):
                navigable_viewpoints.append(viewpoint)
        return navigable_viewpoints


@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spl"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[-1].view_point.position
        )

        if (
            hasattr(task, "is_stop_called") and
            task.is_stop_called and
            distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            ep_success = 1

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance /
            max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._metric is None:
            self._metric = {"count": 0, "is_collision": False}
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True


@registry.register_measure
class DistanceToGoal(Measure):
    """The measure provides a set of metrics that illustrate agent's progress
    towards the goal.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "distance_to_goal"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[0].position
        )
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = {
            "distance_to_target": distance_to_target,
            "start_distance_to_target": self._start_end_episode_distance,
            "distance_delta": self._start_end_episode_distance -
            distance_to_target,
            "agent_path_length": self._agent_episode_distance,
        }


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT"

    def step(
        self,
        *args: Any,
        target: ViewpointData,
        **kwargs: Any
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        position = target.view_point.position
        rotation = target.view_point.rotation

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()

        if kwargs and "episode" in kwargs:
            kwargs["episode"].curr_viewpoint = target.view_point.image_id
            print("Teleporting from %s to %s \n" % (
                 kwargs["episode"].curr_viewpoint,
                 target.view_point.image_id
                 )
             )

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 0:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.MOVE_FORWARD)
        return self._sim.step(HabitatSimActions.MOVE_FORWARD)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 0:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.TURN_LEFT)
        return self._sim.step(HabitatSimActions.TURN_LEFT)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    name: str = "TURN_RIGHT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 0:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.TURN_RIGHT)
        return self._sim.step(HabitatSimActions.TURN_RIGHT)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "STOP"

    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_stop_called = True
        return self._sim.get_observations_at()


@registry.register_task_action
class LookUpAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 0:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.LOOK_UP)
        return self._sim.step(HabitatSimActions.LOOK_UP)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 0:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.LOOK_DOWN)
        return self._sim.step(HabitatSimActions.LOOK_DOWN)


@registry.register_task(name="VLN-v1")
class VLNTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
