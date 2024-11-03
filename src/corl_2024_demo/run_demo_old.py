"""Example of running the phase gait generator."""
from isaacgym import gymapi, gymutil  ## TODO: Only import if running sim. 
from absl import app
from absl import flags

from typing import Sequence

import numpy as np
import os
from rsl_rl.runners import OnPolicyRunner
import time
import torch
import yaml

from src.configs.defaults import sim_config
from src.corl_2023_demo.walk_env import WalkEnv
from src.corl_2023_demo.jump_env import JumpEnv
from src.corl_2023_demo import terrain as trrn
from src.envs import env_wrappers
from src.robots import gamepad_reader
from src.robots.gamepad_reader import ControllerMode
from src.robots import go1, go1_robot
from src.robots.motors import MotorControlMode
from src.utils.torch_utils import to_torch

from ml_collections import ConfigDict

flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to run on real robot.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_string(
    "logdir",
    "corl_demo_ckpts/20231101/3_bound_no_roll_rate/2023_11_01_22_46_14",
    "logdir")
FLAGS = flags.FLAGS


def get_terrain_config():
  config = ConfigDict()

  # config.type = 'plane'
  config.type = 'trimesh'
  config.terrain_length = 10
  config.terrain_width = 10
  config.border_size = 15
  config.num_rows = 1
  config.num_cols = 1
  config.horizontal_scale = 0.05
  config.vertical_scale = 0.005
  config.move_up_distance = 4.5
  config.move_down_distance = 2.5
  config.slope_threshold = 0.75
  config.generation_method = trrn.GenerationMethod.CURRICULUM
  config.max_init_level = 1
  config.terrain_proportions = dict(
      slope_smooth=0.,
      slope_rough=0.,
      stair=0.,
      obstacles=0.,
      stepping_stones=0.,
      gap=0.,
      pit=0.,
      jumping_boxes=1.0
  )
  config.randomize_steps = False
  config.randomize_step_width = True
  # Curriculum setup
  config.curriculum = False
  config.restitution = 0.
  return config

def _create_terrain(gym, sim, terrain_type="plane", terrain_config=None, device="cpu"):
  """Creates terrains.

  Note that we set the friction coefficient to all 0 here. This is because
  Isaac seems to pick the larger friction out of a contact pair as the
  actual friction coefficient. We will set the corresponding friction coef
  in robot friction.
  """
  if terrain_type == 'plane':
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.
    plane_params.dynamic_friction = 1.
    plane_params.restitution = 0.
    gym.add_ground(sim, plane_params)
    terrain = None
  elif terrain_type == 'heightfield':
    terrain = trrn.Terrain(terrain_config,
                                    device=device)
    hf_params = gymapi.HeightFieldParams()
    hf_params.column_scale = terrain_config.horizontal_scale
    hf_params.row_scale = terrain_config.horizontal_scale
    hf_params.vertical_scale = terrain_config.vertical_scale
    hf_params.nbRows = terrain.total_cols
    hf_params.nbColumns = terrain.total_rows
    hf_params.transform.p.x = -terrain_config.border_size
    hf_params.transform.p.y = -terrain_config.border_size
    hf_params.transform.p.z = 0.0
    hf_params.static_friction = 1.
    hf_params.dynamic_friction = 1.
    hf_params.restitution = terrain_config.restitution
    gym.add_heightfield(sim, terrain.height_samples,
                              hf_params)
  elif terrain_type == 'trimesh':
    terrain = trrn.Terrain(terrain_config,
                                    device=device)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]

    tm_params.transform.p.x = -terrain_config.border_size
    tm_params.transform.p.y = -terrain_config.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = 0.99
    tm_params.dynamic_friction = 0.99
    tm_params.restitution = terrain_config.restitution
    gym.add_triangle_mesh(sim,
                                terrain.vertices.flatten(order='C'),
                                terrain.triangles.flatten(order='C'),
                                tm_params)
  else:
    raise ValueError('Invalid terrain type: {}'.format(terrain_type))
  return terrain

def create_sim(sim_conf):
  # from isaacgym import gymapi, gymutil
  gym = gymapi.acquire_gym()
  _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
  if sim_conf.show_gui:
    graphics_device_id = sim_device_id
  else:
    graphics_device_id = -1

  sim = gym.create_sim(sim_device_id, graphics_device_id,
                       sim_conf.physics_engine, sim_conf.sim_params)

  if sim_conf.show_gui:
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1920
    cam_props.height = 1080
    
    viewer = gym.create_viewer(sim, cam_props)
    
    cam_pos = gymapi.Vec3(10.0, 10.0, 3.0) #gymapi.Vec3(5.0, 5.0, 3.0)
    cam_target = gymapi.Vec3(5.0, 5.0, 0.0) #gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
  else:
    viewer = None

  terrain_config = get_terrain_config()
  terrain = _create_terrain(gym, sim, terrain_type="heightfield", terrain_config=terrain_config, device=sim_conf.sim_device)

  # plane_params = gymapi.PlaneParams()
  # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
  # plane_params.static_friction = 1.
  # plane_params.dynamic_friction = 1.
  # plane_params.restitution = 0.
  # gym.add_ground(sim, plane_params)
  return sim, viewer


# def get_init_positions(num_envs: int,
#                        distance: float = 1.,
#                        device: str = "cpu") -> Sequence[float]:
#   num_cols = int(np.sqrt(num_envs))
#   init_positions = np.zeros((num_envs, 3))
#   for idx in range(num_envs):
#     init_positions[idx, 0] = idx // num_cols * distance
#     init_positions[idx, 1] = idx % num_cols * distance
#     init_positions[idx, 2] = 0.26
#   return to_torch(init_positions, device=device)

def get_init_positions(num_envs: int,
                       distance: float = 1.,
                       device: str = "cpu") -> Sequence[float]:
  num_cols = int(np.sqrt(num_envs))
  init_positions = np.zeros((num_envs, 3))
  for idx in range(num_envs):
    # Position robot at x=0 (behind boxes), centered in y, and slightly elevated
    init_positions[idx, 0] = 2  # Start at x=0
    init_positions[idx, 1] = 5  # Center in y-direction (terrain is 10m wide)
    init_positions[idx, 2] = 0.26  # Keep original height
  return to_torch(init_positions, device=device)


def get_latest_policy_path(logdir):
  files = [
      entry for entry in os.listdir(logdir)
      if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
  files = files[::-1]

  for entry in files:
    if entry.startswith("model"):
      return os.path.join(logdir, entry)
  raise ValueError("No Valid Policy Found.")


def get_jump_env(logdir, robot):
  config_path = os.path.join(logdir, "config.yaml")
  policy_path = get_latest_policy_path(logdir)
  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)
  with config.unlocked():
    if FLAGS.use_real_robot:
      config.environment.terrain.type = "plane"
      config.environment.terrain.curriculum = False
    if "turn" not in logdir:
      # Forward Jump, use alternating distances
      config.environment.jumping_distance_schedule = [1.,]
    config.environment.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 6
    config.environment.max_jumps = 5
  env = JumpEnv(robot=robot,
                num_envs=1,
                device="cpu",
                config=config.environment,
                show_gui=FLAGS.show_gui,
                use_real_robot=FLAGS.use_real_robot)
  env = env_wrappers.RangeNormalize(env)
  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
        not FLAGS.use_contact_sensor)

  # Retrieve policy
  runner = OnPolicyRunner(env, config.training, policy_path, device="cpu")
  runner.load(policy_path)
  policy = runner.alg.actor_critic
  return env, policy


def rollout_jump(env, policy, gamepad):
  start_mode = gamepad.mode_command
  state, _ = env.reset()
  policy.reset()
  with torch.no_grad():
    while True:
      action = policy.act_inference(state)
      state, _, _, done, _ = env.step(action)
      if done.any(
      ) or gamepad.estop_flagged or gamepad.mode_command != start_mode:
        break


def main(argv):
  del argv  # unused

  # Initialize Robot
  sim_conf = sim_config.get_config(use_gpu=False,
                                   show_gui=FLAGS.show_gui,
                                   use_real_robot=FLAGS.use_real_robot)
  if not FLAGS.use_real_robot:
    sim, viewer = create_sim(sim_conf)
  else:
    sim, viewer = None, None

  if FLAGS.use_real_robot:
    robot_class = go1_robot.Go1Robot
  else:
    robot_class = go1.Go1

  gamepad = gamepad_reader.Gamepad(vel_scale_x=1.,
                                   vel_scale_y=0.5,
                                   vel_scale_rot=1.,
                                   max_acc=2.)

  robot = robot_class(num_envs=1,
                      init_positions=get_init_positions(
                          1, device=sim_conf.sim_device),
                      sim=sim,
                      viewer=viewer,
                      sim_config=sim_conf,
                      motor_control_mode=MotorControlMode.HYBRID,
                      terrain=None)

  # Initialize Environments
  walk_env = WalkEnv(robot, show_gui=FLAGS.show_gui)
  bound_dir = ("corl_demo_ckpts/20231101/"
               "3_bound_no_roll_rate/2023_11_01_22_46_14")
  bound_env, bound_policy = get_jump_env(bound_dir, robot)
  pronk_dir = ("corl_demo_ckpts/20231103/"
               "1_pronk_forward_relative/2023_11_03_22_20_19")
  pronk_env, pronk_policy = get_jump_env(pronk_dir, robot)
  pronk_turn_dir = ("corl_demo_ckpts/20231103/"
                    "2_pronk_quarter_turn_fixedgait/2023_11_03_22_40_48")
  pronk_turn_env, pronk_turn_policy = get_jump_env(pronk_turn_dir, robot)

  # boxjump_dir = ("corl_demo_ckpts/to_try_minimal/20240802/"
  #                "1_allterrain_regswing_ultimate_hurdledown_distill/2024_08_02_12_22_59")
  # boxjump_env, boxjump_policy = get_jump_env(boxjump_dir, robot)
  # stairjump_dir = ("corl_demo_ckpts/to_try_minimal/20240726/"
  #                   "2_stair_randwidth_pitchperturbation_distill/2024_07_28_13_32_07")
  # stairjump_env, stairjump_policy = get_jump_env(stairjump_dir, robot)

  while True:
    print(f"Current mode: {gamepad.mode_command}")
    gamepad.wait_for_estop_clearance()
    if gamepad.mode_command == ControllerMode.WALK:
      robot.reset()
      while gamepad.mode_command == ControllerMode.WALK and not gamepad.estop_flagged and (not walk_env.is_unsafe().any()):
        walk_env.update_command(*gamepad.speed_command)
        walk_env.step()
    # elif gamepad.mode_command == ControllerMode.BOUND:
    #   rollout_jump(boxjump_env, boxjump_policy, gamepad)
    # elif gamepad.mode_command == ControllerMode.PRONK:
    #   rollout_jump(stairjump_env, stairjump_policy, gamepad)
    elif gamepad.mode_command == ControllerMode.BOUND:
      rollout_jump(bound_env, bound_policy, gamepad)
    elif gamepad.mode_command == ControllerMode.PRONK:
      rollout_jump(pronk_env, pronk_policy, gamepad)
    else:
      rollout_jump(pronk_turn_env, pronk_turn_policy, gamepad)
    gamepad.flag_estop()


if __name__ == "__main__":
  app.run(main)
