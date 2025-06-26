import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose

class GymNavEnv(gym.Env):
    """
    Custom Gymnasium environment wrapping TurtleBot3 in Gazebo via ROS 2.
    Observation: 24-beam LiDAR + (dx, dy) to goal
    Actions: Discrete {forward, left, right}
    """
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        # Assumes rclpy.init() was called externally
        self.node = rclpy.create_node('gym_nav_env')

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(26,), dtype=np.float32
        )

        # ROS publishers and subscribers
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)

        # Internal state
        self.scan = np.ones(24, dtype=np.float32) * 10.0
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.array([2.0, 2.0], dtype=np.float32)
        self._num_boxes_prev = 0

    def _scan_callback(self, msg: LaserScan):
        total = len(msg.ranges)
        step = max(1, total // 24)
        downsampled = msg.ranges[::step][:24]
        cleaned = [r if np.isfinite(r) and r > 0.05 else 10.0 for r in downsampled]
        self.scan = np.array(cleaned, dtype=np.float32)

    def _odom_callback(self, msg: Odometry):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y

    def reset(self, seed=None, options=None):
        # 1) Reset Gazebo physics
        reset_cli = self.node.create_client(Empty, '/reset_simulation')
        if reset_cli.wait_for_service(timeout_sec=3.0):
            fut = reset_cli.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, fut)
        else:
            self.node.get_logger().warn('Reset service unavailable')
        time.sleep(1.0)

        # 2) Teleport robot to start
        state_cli = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        if state_cli.wait_for_service(timeout_sec=3.0):
            state_msg = EntityState()
            state_msg.name = 'burger'
            state_msg.pose.position.x = 0.0
            state_msg.pose.position.y = 0.0
            state_msg.pose.position.z = 0.01
            state_msg.pose.orientation.w = 1.0
            req = SetEntityState.Request()
            req.state = state_msg
            fut = state_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, fut)
        else:
            self.node.get_logger().warn('SetEntityState service unavailable')
        time.sleep(0.5)

        # 3) Delete old dynamic boxes
        delete_cli = self.node.create_client(DeleteEntity, '/delete_entity')
        if delete_cli.wait_for_service(timeout_sec=1.0):
            for i in range(self._num_boxes_prev):
                dr = DeleteEntity.Request()
                dr.name = f"dynamic_box_{i}"
                delete_cli.call_async(dr)
        else:
            self.node.get_logger().warn('DeleteEntity service unavailable')

        # 4) Spawn new dynamic boxes
        spawn_cli = self.node.create_client(SpawnEntity, '/spawn_entity')
        self._num_boxes_prev = 3
        if spawn_cli.wait_for_service(timeout_sec=1.0):
            box_sdf = '''<sdf version="1.6">
  <model name="box">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>'''
            for i in range(self._num_boxes_prev):
                sr = SpawnEntity.Request()
                sr.name = f"dynamic_box_{i}"
                sr.xml = box_sdf
                p = Pose()
                p.position.x = float(np.random.uniform(-1.5, 1.5))
                p.position.y = float(np.random.uniform(-1.5, 1.5))
                p.position.z = 0.25
                sr.initial_pose = p
                spawn_cli.call_async(sr)
        else:
            self.node.get_logger().warn('SpawnEntity service unavailable')
        time.sleep(0.5)

        # 5) Reset internal pose
        self.position[:] = 0.0

        # 6) Return initial observation
        obs = np.concatenate([self.scan, self.goal - self.position])
        obs = np.nan_to_num(obs, nan=10.0, posinf=10.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 10.0).astype(np.float32)
        return obs, {}

    def step(self, action):
        twist = Twist()
        if action == 0:
            twist.linear.x = 0.2
        elif action == 1:
            twist.angular.z = 0.5
        elif action == 2:
            twist.angular.z = -0.5
        self.cmd_pub.publish(twist)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        obs = np.concatenate([self.scan, self.goal - self.position])
        obs = np.nan_to_num(obs, nan=10.0, posinf=10.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 10.0).astype(np.float32)

        dist = np.linalg.norm(self.goal - self.position)
        reward = -dist - 0.1
        terminated = False
        truncated = False

        if np.min(self.scan) < 0.2:
            reward -= 100.0
            terminated = True
        if dist < 0.3:
            reward += 100.0
            terminated = True

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

