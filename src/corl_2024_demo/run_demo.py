"""Evaluate a trained policy."""
from isaacgym import gymapi, gymutil  ## TODO: Only import if running sim. 
from absl import app
from absl import flags

import collections
from datetime import datetime
import os
import pickle
import time

import cv2
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml
from ml_collections import ConfigDict
from typing import Sequence

from src.configs.defaults import sim_config
from src.agents.heightmap_prediction.lstm_heightmap_predictor import LSTMHeightmapPredictor
from src.envs import env_wrappers
from src.envs.terrain import GenerationMethod, Terrain  # pylint: disable=unused-import
from src.robots import gamepad_reader
from src.robots.gamepad_reader import ControllerMode
from src.robots.motors import MotorControlMode
from src.robots import go1, go1_robot
from src.corl_2024_demo.walk_env import WalkEnv
from src.corl_2024_demo.jump_env import JumpEnv
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")

FLAGS = flags.FLAGS


def display_depth(depth_array, depth_cutoff=1):
  # Normalize the depth array to 0-255 for visualization
  depth_array = -np.nan_to_num(depth_array.cpu(), neginf=-depth_cutoff).clip(
      -depth_cutoff, 0.)
  normalized_depth = ((depth_array / depth_cutoff) * 255).astype(np.uint8)
  # Apply colormap
  cv2.imshow("Depth Image", normalized_depth)
  cv2.waitKey(1)  # 1 millisecond delay

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
  config.generation_method = GenerationMethod.CURRICULUM
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
    terrain = Terrain(terrain_config,
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
    terrain = Terrain(terrain_config,
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
  # Load config and policy
  config_path = os.path.join(logdir, "config.yaml")
  policy_path = get_latest_policy_path(logdir)

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    if FLAGS.use_real_robot:
      config.env_config.terrain.type = "plane"
      config.env_config.terrain.curriculum = False
    config.env_config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 6
    config.env_config.max_jumps = 5

  env = JumpEnv(robot=robot,
              num_envs=1,
              device="cpu",
              config=config.env_config,
              show_gui=FLAGS.show_gui,
              use_real_robot=FLAGS.use_real_robot)
  env = env_wrappers.RangeNormalize(env)

  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
        not FLAGS.use_contact_sensor)

  # Initialize teacher policy
  runner = OnPolicyRunner(env,
                          config.teacher_config.training,
                          config.teacher_ckpt,
                          device="cpu")
  runner.load(config.teacher_ckpt)
  policy = runner.alg.actor_critic

  # Retrieve policy
  heightmap_predictor = LSTMHeightmapPredictor(
      dim_output=len(config.env_config.measured_points_x) *
      len(config.env_config.measured_points_y),
      vertical_res=config.env_config.camera_config.vertical_res,
      horizontal_res=config.env_config.camera_config.horizontal_res,
  ).to("cpu")
  heightmap_predictor.load(policy_path)
  heightmap_predictor.eval()
  return env, policy, heightmap_predictor

def get_camera_config():
  config = ConfigDict()

  config.horizontal_fov_deg = 21.43
  config.vertical_fov_deg = 59.18
  config.horizontal_res = 16
  config.vertical_res = 48
  config.position_in_base_frame = [0.245 + 0.027, 0.0075, 0.072 + 0.02]
  config.orientation_rpy_in_base_frame = [0., 0.52, 0.]
  return config

def rollout_jump(env, policy, heightmap_predictor, gamepad, image_buffer=None):
  start_mode = gamepad.mode_command
  env.reset()
  policy.reset()
  heightmap_predictor.reset()
  
  # Handle different image collection for sim vs real
  if not FLAGS.use_real_robot:
    images = collections.deque(maxlen=5)
  
  with torch.no_grad():
    while True:
      if FLAGS.use_real_robot:
        curr_imgs = image_buffer.last_image
      else:
        curr_imgs = []
        for env_id in range(1):
          curr_imgs.append(to_torch(env.robot.get_camera_image(env_id, mode="depth"), device="cpu"))
        curr_imgs = torch.stack(curr_imgs, dim=0)
        images.append(curr_imgs)
        curr_imgs = images[0]

      proprioceptive_state = env.get_proprioceptive_observation()
      height = heightmap_predictor.forward(
          base_state=env.get_perception_base_states(),
          foot_positions=(
              env.robot.foot_positions_in_gravity_frame[:, :, [0, 2]] *
              env.gait_generator.desired_contact_state_state_estimation[:, :,
                                                                        None]
          ).reshape((-1, 8)),
          depth_image=curr_imgs)
      obs = torch.concatenate((proprioceptive_state, height), dim=-1)
      normalized_obs = env.normalize_observ(obs)

      action = policy.act_inference(normalized_obs)
      action = action.clip(min=env.action_space[0], max=env.action_space[1])
      display_depth(curr_imgs[0] if not FLAGS.use_real_robot else curr_imgs)
      _, _, reward, done, info = env.step(
            action, base_height_override=-height[:, 10])
      if done.any(
      ) or gamepad.estop_flagged or gamepad.mode_command != start_mode:
        break

def main(argv):
  del argv  # unused

  # Initialize ROS node for real robot
  if FLAGS.use_real_robot:

    # Add imports for real robot
    from sensor_msgs.msg import Image
    import rospy
    from cv_bridge import CvBridge

    # Buffer to store depth image embedding for real robot
    class DepthImageBuffer:
      """Buffer to store depth image embedding for real robot."""
      def __init__(self):
        self._bridge = CvBridge()
        self._last_image = torch.zeros((1, 48, 60))
        self._last_image_time = time.time()

      def update_image(self, msg):
        frame = np.array(self._bridge.imgmsg_to_cv2(msg))
        self._last_image = torch.from_numpy(frame)[None, ...]
        self._last_image_time = time.time()

      @property
      def last_image(self):
        return self._last_image

      @property
      def last_image_time(self):
        return self._last_image_time
    
    rospy.init_node("run_demo")
    image_buffer = DepthImageBuffer()
    rospy.Subscriber("/camera/depth/cnn_input",
                     Image,
                     image_buffer.update_image,
                     queue_size=1,
                     tcp_nodelay=True)
  else:
    image_buffer = None

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

  robot = robot_class(
        num_envs=1,
        init_positions=get_init_positions(
                          1, device=sim_conf.sim_device),
        sim=sim,
        viewer=viewer,
        sim_config=sim_conf,
        motor_control_mode=MotorControlMode.HYBRID,
        terrain=None,
        motor_torque_delay_steps=5,
        camera_config=get_camera_config(),
        randomize_com=False)
  
  # Initialize Environments
  walk_env = WalkEnv(robot, show_gui=FLAGS.show_gui)
  boxjump_dir = ("data/box_policy_20240802/"
                 "box_distill")
  boxjump_env, boxjump_policy, boxjump_heightmap_predictor = get_jump_env(boxjump_dir, robot)

  while True:
    print(f"Current mode: {gamepad.mode_command}")
    gamepad.wait_for_estop_clearance()
    if gamepad.mode_command == ControllerMode.WALK:
      robot.reset()
      while gamepad.mode_command == ControllerMode.WALK and not gamepad.estop_flagged and (not walk_env.is_unsafe().any()):
        walk_env.update_command(*gamepad.speed_command)
        walk_env.step()
    elif gamepad.mode_command == ControllerMode.BOUND:
      rollout_jump(boxjump_env, boxjump_policy, boxjump_heightmap_predictor, gamepad, image_buffer)
    gamepad.flag_estop()


if __name__ == "__main__":
  app.run(main)
