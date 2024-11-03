"""Policy outputs desired CoM speed for Go1 to track the desired speed."""
# pytype: disable=attribute-error
import ml_collections
import numpy as np
import torch

from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.robots import go1_robot


def get_gait_config():
  config = ml_collections.ConfigDict()
  config.stepping_frequency = 2.5
  config.initial_offset = np.array([0., 0.5, 0.5, 0.],
                                   dtype=np.float32) * (2 * np.pi)
  config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
  return config


class WalkEnv:
  """Interface for Walking envionment."""
  def __init__(self, robot, show_gui):
    self._robot = robot
    self._show_gui = show_gui
    gait_config = get_gait_config()
    self._gait_generator = phase_gait_generator.PhaseGaitGenerator(
        self._robot, gait_config)
    self._swing_leg_controller = raibert_swing_leg_controller.\
      RaibertSwingLegController(
          self._robot,
          self._gait_generator,
          foot_landing_clearance=0.,
          foot_height=0.1)
    self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        self._robot,
        self._gait_generator,
        desired_body_height=0.26,
        weight_ddq=np.diag([1., 1., 10., 10., 10., 1.]),
        body_inertia=np.array([0.14, 0.35, 0.35]) * 4,
        foot_friction_coef=0.4,
        use_full_qp=True,
    )

  def update_command(self, lin_command, ang_command):
    self._torque_optimizer.desired_linear_velocity = torch.Tensor(lin_command).unsqueeze(0)
    self._torque_optimizer.desired_angular_velocity = torch.Tensor([0., 0., ang_command]).unsqueeze(0)

  def step(self):
    if isinstance(self._robot, go1_robot.Go1Robot):
      self._robot.state_estimator.update_foot_contact(
          self._gait_generator.desired_contact_state)
    self._gait_generator.update()
    self._swing_leg_controller.update()

    motor_action, _, _, _, _ = self._torque_optimizer.get_action(
        self._gait_generator.desired_contact_state,
        swing_foot_position=self._swing_leg_controller.desired_foot_positions)
    self._robot.step(motor_action)

    if self._show_gui:
      self._robot.render()

  def is_unsafe(self):
    return torch.logical_or(self._robot.projected_gravity[:, 2] < 0.5,
                            self._robot.base_position[:, 2] < 0.12)
