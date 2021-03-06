#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import attr
import numpy as np
import quaternion

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
    Simulator,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector
)
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    dir_angle_between_quaternions,
)

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
    mask: Optional[List[int]] = None
    tokens_length: int = 0


@attr.s(auto_attribs=True, kw_only=True)
class ViewpointData:
    r"""Base class for a viewpoint specification.
    """
    image_id: int = attr.ib(default=None, validator=not_none_validator)
    view_point: AgentState = None
    radius: Optional[float] = None

    def __str__(self):
        return str(self.image_id)

    def __repr__(self):
        return "image_id {0} \n position {1} \n rotation {2} \n".format(
            self.image_id,
            self.view_point.position,
            self.view_point.rotation
        )

    def get_position(self):
        return self.view_point.position

    def get_rotation(self):
        return self.view_point.rotation


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
    curr_viewpoint: Optional[ViewpointData] = None
    distance = None

    def reset(self):
        self.curr_viewpoint = self.goals[0]


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
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

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
class ElevationSensor(Sensor):
    r"""Sensor for observing the agent's elevation in the global coordinate
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
        return "elevation"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NULL

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_rot = agent_state.rotation
        camera_rot = agent_state.sensor_states["rgb"].rotation

        return dir_angle_between_quaternions(agent_rot, camera_rot)


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
        self.max_locations = 20
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "adjacentViewpoints"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NULL  # Missing sensor type

    def normalize_angle(self, angle):
        # Matterport goes from 0 to 2pi going clock wise.
        # Habitat goes from 0 - pi going counter clock wise.
        # Also habitat goes from 0 to - pi clock wise.
        # This method normalizes to Matterport heading format.
        if 0 <= angle < np.pi:
            return 2 * np.pi - angle
        return -angle

    def get_rel_heading(self, posA, rotA, posB):

        direction_vector = np.array([0, 0, -1])
        quat = quaternion_from_coeff(rotA).inverse()

        # The heading vector and heading angle are in arctan2 format
        heading_vector = quaternion_rotate_vector(quat, direction_vector)
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        heading = self.normalize_angle(heading)

        adjusted_heading = np.pi/2 - heading
        camera_horizon_vec = [
            np.cos(adjusted_heading),
            np.sin(adjusted_heading),
            0
        ]

        # This vectors are in habitat format we need to rotate them.
        rotated_posB = [posB[0], -posB[2], posB[1]]
        rotated_posA = [posA[0], -posA[2], posA[1]]
        target_vector = np.array(rotated_posB) - np.array(rotated_posA)

        y = target_vector[0] * camera_horizon_vec[1] - \
            target_vector[1] * camera_horizon_vec[0]
        x = target_vector[0] * camera_horizon_vec[0] + \
            target_vector[1] * camera_horizon_vec[1]

        return np.arctan2(y, x)

    def get_rel_elevation(self, posA, rotA, cameraA, posB):
        direction_vector = np.array([0, 0, -1])
        quat = quaternion_from_coeff(rotA)
        rot_vector = quaternion_rotate_vector(quat.inverse(), direction_vector)

        camera_quat = quaternion_from_coeff(cameraA)
        camera_vector = quaternion_rotate_vector(
            camera_quat.inverse(),
            direction_vector
        )

        elevation_angle = dir_angle_between_quaternions(quat, camera_quat)

        rotated_posB = [posB[0], -posB[2], posB[1]]
        rotated_posA = [posA[0], -posA[2], posA[1]]
        target_vector = np.array(rotated_posB) - np.array(rotated_posA)
        target_z = target_vector[2]
        target_length = np.linalg.norm([target_vector[0], target_vector[1]])

        rel_elevation = np.arctan2(target_z, target_length)
        return rel_elevation - elevation_angle

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        # This number is the maximum connections per node
        sensor_shape = (self.max_locations, 18)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def pad_locations(self, nav_locations):
        pad = [-1] * 18
        seq = [pad] * (self.max_locations - len(nav_locations))
        nav_locations.extend(seq)
        return nav_locations

    def format_location(
        self,
        restricted,
        image_id,
        rel_heading,
        rel_elevation,
        target_pos,
        agent_rot,
        camera_pos,
        camera_rot,
    ):
        '''
        Format:
        [restricted, image_id, rel_heading, rel_elevation,
            x, y, z, rx, ry, rz, rw, cx, cy, cz, crx, cry, crz, crw]

        restricted = True if the viewpoint is within the FOV
        image_id: int = viewpoint id
        rel_heading: float = relative heading from the initial point (radians)
        rel_elevation: float = relative elevation from the initial point (radians)
        x, y, z: float = position coordinates
        rx, ry, rz, rw = quaternion values
        cx, cy, cz: float = position coordinates of the camera
        crx, cry, crz, crw = quaternion values of the camera
        '''
        formatted_location = [restricted, image_id, rel_heading, rel_elevation]
        formatted_location.extend(target_pos)
        formatted_location.extend(agent_rot)
        formatted_location.extend(camera_pos)
        formatted_location.extend(camera_rot)

        return formatted_location


    def get_observation(
        self, observations, episode, task, *args: Any, **kwargs: Any
    ):
        scan = episode.scan
        curr_viewpoint = episode.curr_viewpoint.image_id

        near_viewpoints = \
            task.get_navigable_locations(scan, curr_viewpoint)

        agent_state = self._sim.get_agent_state()

        agent_pos = agent_state.position
        camera_rot = quaternion_to_list(agent_state.sensor_states["rgb"].rotation)
        camera_pos = agent_state.sensor_states["rgb"].position
        agent_rot = quaternion_to_list(agent_state.rotation)
        angle = self._sim.config.RGB_SENSOR.HFOV * 2 * np.pi / 360 / 2

        navigable_viewpoints = [
            self.format_location(
                1,
                curr_viewpoint,
                0,
                0,
                agent_pos,
                agent_rot,
                camera_pos,
                camera_rot,
            )
        ]

        for viewpoint in near_viewpoints:
            image_id = viewpoint["image_id"]
            target_pos = viewpoint["start_position"]
            rel_heading = self.get_rel_heading(
                    agent_pos,
                    agent_rot,
                    target_pos
            )
            rel_elevation = self.get_rel_elevation(
                    agent_pos,
                    agent_rot,
                    camera_rot,
                    target_pos
            )
            restricted = 1

            if -angle <= rel_heading <= angle:
                restricted = 0

            navigable_viewpoints.append(
                self.format_location
                (
                    restricted,
                    image_id,
                    rel_heading,
                    rel_elevation,
                    target_pos,
                    agent_rot,
                    camera_pos,
                    camera_rot,
                )
            )

        return self.pad_locations(navigable_viewpoints)


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

    def _episode_distance_from_path(self, episode: VLNEpisode):
        previous_goal = episode.goals[0]
        total_distance = 0.0
        for goal in episode.goals[1:]:
            posA = previous_goal.get_position()
            posB = goal.get_position()
            total_distance += self._euclidean_distance(posA, posB)
            previous_goal = goal
        return total_distance

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        if episode.distance:
            self._start_end_episode_distance = episode.distance
        else:
            self._start_end_episode_distance = \
                self._episode_distance_from_path(episode)
        #self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        position_a[1]=0
        position_b[1]=0
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a)#, ord=2
        )

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        discrete = getattr(task, "is_discrete")
        if discrete:
            curr_viewpoint_id = episode.curr_viewpoint.image_id
            goal = episode.goals[-1].image_id
            scan = episode.scan
            distance_to_target = \
                task.get_distance_to_target(scan, curr_viewpoint_id, goal)
        else:
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[-1].get_position()
            )

            if np.isinf(distance_to_target):
                print(
                    "WARNING: The Success metric might be compromised " +
                    "The geodesic_distance failed " +
                    "looking for a snap_point instead"
                )
                new_position = np.array(current_position, dtype='f')
                new_position = self._sim._sim.pathfinder.snap_point(
                                new_position
                               )
                if np.isnan(new_position[0]):
                    print(
                        "ERROR: The Success metric is compromised " +
                        "The geodesic_distance failed " +
                        "Cannot find path"
                    )
                else:
                    current_position = new_position
                    distance_to_target = self._sim.geodesic_distance(
                        current_position, episode.goals[-1].get_position()
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

        # removing rounding error of 0.0000025 meters
        if self._metric >= 0.9999975:
            self._metric = 1.0


@registry.register_measure
class Success(Measure):

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "success"


    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        discrete = getattr(task, "is_discrete")
        if discrete:
            curr_viewpoint_id = episode.curr_viewpoint.image_id
            goal = episode.goals[-1].image_id
            scan = episode.scan
            distance_to_target = \
                task.get_distance_to_target(scan, curr_viewpoint_id, goal)
        else:
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[-1].get_position()
            )

            if np.isinf(distance_to_target):
                print(
                    "WARNING: The Success metric might be compromised " +
                    "The geodesic_distance failed " +
                    "looking for a snap_point instead"
                )
                new_position = np.array(current_position, dtype='f')
                new_position = self._sim._sim.pathfinder.snap_point(
                                new_position
                               )
                if np.isnan(new_position[0]):
                    print(
                        "ERROR: The Success metric is compromised " +
                        "The geodesic_distance failed " +
                        "Cannot find path"
                    )
                else:
                    current_position = new_position
                    distance_to_target = self._sim.geodesic_distance(
                        current_position, episode.goals[-1].get_position()
                    )

        if (
            hasattr(task, "is_stop_called") and
            task.is_stop_called and
            distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            ep_success = 1

        self._metric = ep_success


@registry.register_measure
class OracleSuccess(Measure):

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._nearest_distance = -1

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracleSuccess"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None
        self._nearest_distance = -1

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = 0

        discrete = getattr(task, "is_discrete")
        if discrete:
            curr_viewpoint_id = episode.curr_viewpoint.image_id
            goal = episode.goals[-1].image_id
            scan = episode.scan
            distance_to_target = \
                task.get_distance_to_target(scan, curr_viewpoint_id, goal)
        else:
            current_position = self._sim.get_agent_state().position.tolist()
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[-1].get_position()
            )
            if np.isinf(distance_to_target):
                print(
                    "WARNING: The Oracle distance might be compromised " +
                    "The geodesic_distance failed " +
                    "looking for a snap_point instead"
                )
                new_position = np.array(current_position, dtype='f')
                new_position = self._sim._sim.pathfinder.snap_point(
                                new_position
                               )
                if np.isnan(new_position[0]):
                    print(
                        "ERROR: The Oracle distance is compromised " +
                        "The geodesic_distance failed " +
                        "Cannot find path"
                    )
                else:
                    current_position = new_position
                    distance_to_target = self._sim.geodesic_distance(
                        current_position, episode.goals[-1].get_position()
                    )
        if (
            self._nearest_distance == -1 or
            distance_to_target < self._nearest_distance
        ):
            self._nearest_distance = distance_to_target

        if (
            hasattr(task, "is_stop_called") and
            task.is_stop_called and
            self._nearest_distance < self._config.ORACLE_SUCCESS_DISTANCE
        ):
            ep_success = 1

        self._metric = ep_success


@registry.register_measure
class TrajectoryLength(Measure):

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._previous_position = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "trajectoryLength"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._metric = 0.0

    def _euclidean_distance(self, position_a, position_b):
        position_a[1]=0  # project to xy plane
        position_b[1]=0  # project to xy plane
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a)
        )

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        self._metric += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position


@registry.register_measure
class NavigationError(Measure):
    r"""
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "navigationError"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0.0

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        discrete = getattr(task, "is_discrete")
        if discrete:
            curr_viewpoint_id = episode.curr_viewpoint.image_id
            goal = episode.goals[-1].image_id
            scan = episode.scan
            distance_to_target = \
                task.get_distance_to_target(scan, curr_viewpoint_id, goal)
        else:
            current_position = self._sim.get_agent_state().position.tolist()
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[-1].get_position()
            )
            if np.isinf(distance_to_target):
                print(
                    "WARNING: The Oracle distance might be compromised " +
                    "The geodesic_distance failed " +
                    "looking for a snap_point instead"
                )
                new_position = np.array(current_position, dtype='f')
                new_position = self._sim._sim.pathfinder.snap_point(
                                new_position
                               )
                if np.isnan(new_position[0]):
                    print(
                        "ERROR: The Oracle distance is compromised " +
                        "The geodesic_distance failed " +
                        "Cannot find path"
                    )
                else:
                    current_position = new_position
                    distance_to_target = self._sim.geodesic_distance(
                        current_position, episode.goals[-1].get_position()
                    )

        self._metric = distance_to_target


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
        if episode.distance:
            self._start_end_episode_distance = episode.distance
        else:
            self._start_end_episode_distance = self._sim.geodesic_distance(
                self._previous_position, episode.goals[-1].get_position()
            )
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        position_a[1]=0
        position_b[1]=0
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a)#, ord=2
        )

    def update_metric(self, episode, task, action, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[-1].get_position()
        )

        start = episode.curr_viewpoint.image_id
        end = episode.goals[-1].image_id
        distance_to_target = task.get_distance_to_target(
            episode.scan,
            start,
            end
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

        #print("Teleport Action to pos ", target.view_point.position)
        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            #print("The destination is not navigable, running snap point")
            # is not navigable then we search for a location close to the target
            new_position = np.array(position, dtype='f')
            new_position = self._sim._sim.pathfinder.snap_point(new_position)
            if np.isnan(new_position[0]):
                #print("Snap point couldn't find a place to land, error.")
                return self._sim.get_observations_at()
            else:
                position = new_position#.tolist()
                #print("New position found", position)

        if kwargs and "episode" in kwargs:
            last_viewpoint = kwargs["episode"].curr_viewpoint
            kwargs["episode"].curr_viewpoint = ViewpointData(
                            image_id=target.image_id,
                            view_point=AgentState(
                                position=position,
                                rotation=rotation)
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
        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 1:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.MOVE_FORWARD)
        return self._sim.step(HabitatSimActions.MOVE_FORWARD)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    name: str = "TURN_LEFT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        # Right and left are inverted in MatterportSIM ???
        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 1:
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

        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 1:
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

        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 1:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.LOOK_UP)
        return self._sim.step(HabitatSimActions.LOOK_UP)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if kwargs and 'num_steps' in kwargs and kwargs['num_steps'] > 1:
            for _ in range(kwargs["num_steps"] - 1):
                self._sim.step(HabitatSimActions.LOOK_DOWN)
        return self._sim.step(HabitatSimActions.LOOK_DOWN)


@registry.register_task(name="VLN-v1")
class VLNTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        self.is_discrete = getattr(config, 'DISCRETE')
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def get_distance_to_target(self, scan, start_viewpoint, end_viewpoint):
        if self._dataset:
            return self._dataset.get_distance_to_target(
                scan,
                start_viewpoint,
                end_viewpoint
            )
        return float("inf")

    def get_shortest_path_to_target(self, scan, start_viewpoint, end_viewpoint):
            if self._dataset:
                return self._dataset.get_shortest_path_to_target(
                    scan,
                    start_viewpoint,
                    end_viewpoint
                )
            return []

    def get_navigable_locations(self, scan, viewpoint):
        if self._dataset:
            return self._dataset.get_navigable_locations(scan, viewpoint)
        return {}

    def get_action_tokens(self):
        return self._dataset.get_action_tokens()
