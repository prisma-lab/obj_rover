#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from obj_msgs.msg import DetectedObjectsList
from rclpy.action import ActionClient
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from obj_agent.agents import *
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import math
import time
import random

save_log = False
log_dict = {'map_step': [], 'fov_step': [], 'attention_maps_step': [], 'attention_weights_step': []}

class KalmanFilterPositionOnly:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.F = np.eye(2)
        self.H = np.eye(2)
        self.Q = process_variance * np.eye(2)
        self.R = measurement_variance * np.eye(2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.reshape(z, (2, 1))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()


class AgentNode(Node):

    def __init__(self, tasks, use_learned_policy=False):
        super().__init__('agent_node')
        self.get_logger().info('Initializing agent node...')

        callback_group = ReentrantCallbackGroup()

        # Variable initialization
        self.custom_resolution = 0.43  # meters per "step" (grid cell size)
        self.objects = []
        # self.locations = [] # Unused
        # self.obj = [] # Unused
        self.map = None
        self.resized_map = None
        self.origin = None
        self.init_count = 0

        # Pose from topic (meters + quaternion z,w)
        self.pose_from_topic = None  # [x, y, z, w]
        self.direction = None

        self.map_ready = False

        self.is_navigating = False
        self.kalman_init_steps = 1#3
        self.tasks = tasks
        self.mission = tasks[0]#'go to the green box'
        print(f"Mission: {self.mission}")
        self.mission_index = 0

        self.CLASS_TO_IDX = {"toolbox": 7, "spray can": 6}
        self.COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "gray": 5}
        self.IDX_TO_COLOR = dict(zip(self.COLOR_TO_IDX.values(), self.COLOR_TO_IDX.keys()))
        self.real_map_grid_mapping = {
            0: 1,     # MiniGrid empty
            100.: 2,  # MiniGrid wall
            -1: 0     # MiniGrid empty for now, to clarify
        }
        self.objects_symbols = {5: "|", 6: "o", 7: "s"}
        self.view_size = 7

        self.bridge = CvBridge()

        self.device = "cpu"
        self.agent = Agent().to(self.device)
        self.agent.load_state_dict(
            torch.load(
                "/home/user/ros2_ws/src/pkg/obj_agent/obj_agent/single_stream_agent_1752772241.cleanrl_model",
                map_location=self.device
            ),
            strict=False
        )
        self.agent.eval()

        self.vocab = Vocabulary(vocab_max_size)
        self.vocab.load_from_json("/home/user/ros2_ws/src/pkg/obj_agent/obj_agent/single_stream_vocab.json")

        self.next_lstm_state = (
            torch.zeros(self.agent.internal_memory.num_layers, 1, self.agent.internal_memory.lstm_emb_size).to(self.device),
            torch.zeros(self.agent.internal_memory.num_layers, 1, self.agent.internal_memory.lstm_emb_size).to(self.device),
        )

        qos_profile_image = QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=6, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Map and objects subscriptions
        self.obj_sub = self.create_subscription(
            DetectedObjectsList, 'objects_detected', self.objects_callback, 1,
            callback_group=callback_group
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'rover/map', self.map_callback, 1,
            callback_group=callback_group
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, 'rover_tf_pose', self.pose_callback, 1,
            callback_group=callback_group
        )
        self.grid_pub = self.create_publisher(Image, 'grid_map', qos_profile_image)

        self.timer = self.create_timer(1.0, self.agent_call, callback_group=callback_group)

        # Move base client
        self.client = ActionClient(self, NavigateToPose, 'rover/navigate_to_pose')

        # Visualization
        self.visualization = True

        self.use_learned_policy = use_learned_policy

        if self.visualization:
            self.fig, self.ax = plt.subplots()
            self.fig.set_figwidth(15)
            self.fig.set_figheight(15)

        self.get_logger().info('Done!')

    # -------------------------
    # Helpers
    # -------------------------
    def _yaw_from_zw(self, z, w):
        return 2.0 * math.atan2(z, w)

    def _wrap_to_pi(self, a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _quantize_yaw_90(self, yaw):
        candidates = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        yaw = self._wrap_to_pi(yaw)
        best = min(candidates, key=lambda c: abs(self._wrap_to_pi(yaw - c)))
        return best

    def _quat_zw_from_yaw(self, yaw):
        yaw = self._wrap_to_pi(yaw)
        z = math.sin(yaw * 0.5)
        w = math.cos(yaw * 0.5)
        return z, w

    def serialize_object(self, obj):
        # NOTE: This returns position in METERS from the ROS message
        return {
            'cls': str(obj.cls),
            'trackID': int(obj.trackid),
            'pos2D': [obj.pos2d.points[0].x, obj.pos2d.points[0].y] 
        }

    def meters_to_px(self, coord_m, ori, h):
        # Converte metri -> pixel griglia corrente
        return [
            (
                round((y - ori[1]) / self.custom_resolution),
                round(h - ((x - ori[0]) / self.custom_resolution))
            )
            for x, y in coord_m
        ]

    def refresh_object_pixel_positions(self):
        """
        Ricalcola la posizione pos2D (pixel) di tutti gli oggetti 
        basandosi sulla loro real_pos (metri) e sulla mappa attuale.
        Da chiamare quando la mappa cambia o prima di generare l'osservazione.
        """
        if self.resized_map is None or self.origin is None:
            return

        h = self.resized_map.shape[1] # Nota: shape[1] è width nel tuo uso di meters_to_px, verifica orientamento
        # In meters_to_px usi shape[1] come altezza logica per l'inversione asse X.
        
        for obj in self.objects:
            if "real_pos" in obj:
                # real_pos è (2,) numpy array o list
                px_coords = self.meters_to_px([obj["real_pos"]], self.origin, h)
                # Aggiorna il valore pixel per la visualizzazione/logica corrente
                obj["pos2D"] = (px_coords[0][1], px_coords[0][0])

    # -------------------------
    # Callbacks
    # -------------------------
    def set_direction(self):
        if self.pose_from_topic is None:
            return
        z = self.pose_from_topic[2]
        w = self.pose_from_topic[3]
        yaw = self._yaw_from_zw(z, w)

        if yaw > np.radians(135) or yaw < np.radians(-135):
            self.direction = 'right'
        elif yaw > np.radians(-135) and yaw < np.radians(-45):
            self.direction = 'up'
        elif yaw > np.radians(-45) and yaw < np.radians(45):
            self.direction = 'left'
        elif yaw > np.radians(45) and yaw < np.radians(135):
            self.direction = 'down'

    def pose_callback(self, msg):
        self.pose_from_topic = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        self.set_direction()

    def objects_callback(self, msg):
        # Non processare se non abbiamo una mappa (perchè non sapremmo convertirli in pixel inizialmente)
        # Ma è meglio salvare il dato grezzo in metri comunque.
        if not self.map:
            return

        for obj in msg.objects:
            obj_ = self.serialize_object(obj) # pos2D qui è in METRI
            current_pos_meters = obj_["pos2D"]
            
            distances = np.linalg.norm(
                np.array(current_pos_meters) - np.array([o["real_pos"] for o in self.objects]),
                axis=1
            ) if self.objects else np.array([])

            if distances.size == 0 or not np.any(distances < self.custom_resolution):
                # Nuovo oggetto
                new_obj = {
                    "id": self.CLASS_TO_IDX[obj_['cls']],
                    "color": obj_['trackID'],
                    'kf': KalmanFilterPositionOnly(),
                    'detection_steps': 1,
                    'real_pos': current_pos_meters # IMPORTANTE: salviamo la posizione reale
                }
                # Inizializza KF con misura in metri
                new_obj['kf'].predict()
                new_obj['kf'].update(current_pos_meters)
                self.objects.append(new_obj)
            else:
                # Aggiorna oggetto esistente
                index = np.argmin(distances)
                self.objects[index]["detection_steps"] += 1
                self.objects[index]["kf"].predict()
                self.objects[index]["kf"].update(current_pos_meters)
                # Aggiorna la posizione reale stimata dal filtro
                self.objects[index]["real_pos"] = self.objects[index]["kf"].get_state()

            # NOTA: Non calcoliamo "pos2D" (pixel) qui. 
            # Lo faremo on-demand in refresh_object_pixel_positions() 
            # per garantire che sia coerente con la mappa corrente.

    def map_callback(self, msg):
        self.map = msg
        self.map_processing()
        # Appena la mappa cambia, aggiorniamo i pixel degli oggetti per sicurezza
        if self.resized_map is not None:
            self.refresh_object_pixel_positions()

    def map_processing(self):
        if self.map and self.is_navigating is False:
            width = self.map.info.width
            height = self.map.info.height
            self.origin = [self.map.info.origin.position.x, self.map.info.origin.position.y]
            occupancy_data = self.map.data
            map_matrix = np.array(occupancy_data).reshape((height, width))

            occ_mask = (map_matrix == 100).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilated_mask = cv2.dilate(occ_mask, kernel)
            map_matrix[dilated_mask == 1] = 100

            scale_factor = self.map.info.resolution / self.custom_resolution
            new_size = (
                int(map_matrix.shape[1] * scale_factor),
                int(map_matrix.shape[0] * scale_factor)
            )
            self.resized_map = cv2.resize(map_matrix, new_size, interpolation=cv2.INTER_NEAREST)

    def policy_movement(self, command, cur_yaw_q, cur_x_m, cur_y_m, step):
        # 1. Gestione Orientamento (Goal Yaw)
        goal_yaw = cur_yaw_q
        
        if command == 1:      # right
            self.get_logger().info('Turning right...')
            goal_yaw = self._wrap_to_pi(cur_yaw_q - math.pi / 2.0)
        elif command == 0:    # left
            self.get_logger().info('Turning left...')
            goal_yaw = self._wrap_to_pi(cur_yaw_q + math.pi / 2.0)
        elif command == 2:    # forward
            goal_yaw = cur_yaw_q

        goal_z, goal_w = self._quat_zw_from_yaw(goal_yaw)

        # 2. Gestione Posizione (Goal X, Y)
        goal_x = cur_x_m
        goal_y = cur_y_m

        if command == 2:  # FORWARD
            self.get_logger().info(f'Moving forward ({self.direction})...')

            if self.resized_map is None or self.origin is None:
                self.get_logger().warn("Mappa non pronta.")
                return -1

            map_width_px = self.resized_map.shape[1]

            cur_grid_y = round(-self.origin[1] / self.custom_resolution + cur_y_m / self.custom_resolution)
            cur_grid_x = round((map_width_px + self.origin[0] / self.custom_resolution) - cur_x_m / self.custom_resolution)

            target_grid_x = cur_grid_x
            target_grid_y = cur_grid_y

            if self.direction == 'right':
                target_grid_x += 1
            elif self.direction == 'left':
                target_grid_x -= 1
            elif self.direction == 'up':
                target_grid_y -= 1
            elif self.direction == 'down':
                target_grid_y += 1

            goal_y = target_grid_y * self.custom_resolution + self.origin[1]
            goal_x = (map_width_px * self.custom_resolution + self.origin[0]) - (target_grid_x * self.custom_resolution)

            self.get_logger().info(f"Grid: [{cur_grid_x},{cur_grid_y}] -> [{target_grid_x},{target_grid_y}]")

        elif command == 6: # Done
            self.get_logger().info('Mission Done')
            self.reset_agent_memory()
            if self.mission_index < len(self.tasks) - 1:
                self.mission_index += 1
                self.mission = self.tasks[self.mission_index]
                print(f"Mission: {self.mission}")
            else:
                self.mission = input("Enter new mission: ")
                print(self.mission)
            return -1
        
        elif command not in [0, 1, 2]: # Invalid
            return -1
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_x)
        goal_msg.pose.pose.position.y = float(goal_y)
        goal_msg.pose.pose.orientation.z = float(goal_z)
        goal_msg.pose.pose.orientation.w = float(goal_w)

        self.is_navigating = True
        self.client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)
        self.map_ready = False
        return 1

    def agent_call(self):
        #print(self.direction)
        if not self.direction:
            return

        if self.map and (self.is_navigating is False):
            # time.sleep(1) # Rimosso per fluidità, se necessario riabilitalo
            if self.pose_from_topic is None:
                return
            if self.origin is None or self.resized_map is None:
                return

            self.get_logger().info('Calculating step...')

            # 1. AGGIORNAMENTO CRUCIALE: Ricalcola i pixel degli oggetti sulla mappa ATTUALE
            self.refresh_object_pixel_positions()

            # Use real pose (meters)
            cur_x_m = float(self.pose_from_topic[0])
            cur_y_m = float(self.pose_from_topic[1])
            cur_yaw = self._yaw_from_zw(self.pose_from_topic[2], self.pose_from_topic[3])
            cur_yaw_q = self._quantize_yaw_90(cur_yaw)

            # Build grid for observation
            robot_px = [
                round(-self.origin[1] / self.custom_resolution + cur_y_m / self.custom_resolution),
                round((self.resized_map.shape[1] + self.origin[0] / self.custom_resolution) - cur_x_m / self.custom_resolution)
            ]

            flipped_map = np.flip(self.resized_map, 1)
            flipped_map[flipped_map == 0.0] = 50
            flipped_map[flipped_map == -1.0] = 0

            agent_y, agent_x = robot_px
            # ... (logica di visualizzazione rot e marker invariata) ...
            marker = ""
            rot = 0
            offset = self.view_size // 2

            if self.direction == 'right':
                view_x = agent_x
                view_y = agent_y - offset
                marker = ">"
                rot = 2
            elif self.direction == 'left':
                view_x = agent_x - self.view_size + 1
                view_y = agent_y - offset
                marker = "<"
                rot = 0
            elif self.direction == 'down':
                view_x = agent_x - offset
                view_y = agent_y
                marker = "v"
                rot = -1
            elif self.direction == 'up':
                view_x = agent_x - offset
                view_y = agent_y - self.view_size + 1
                marker = "^"
                rot = 1

            sub_matrix = np.zeros((self.view_size, self.view_size))
            obj_pos_sub_matrix = np.zeros((self.view_size, self.view_size))
            objects_in_fov = []

            obj_fov_id = 1
            for obj in self.objects:
                # obj["pos2D"] ora è garantito essere aggiornato rispetto alla mappa corrente
                if (
                    obj["detection_steps"] >= self.kalman_init_steps and
                    obj["pos2D"][1] - view_y >= 0 and
                    obj["pos2D"][0] - view_x >= 0 and
                    obj["pos2D"][1] - view_y < self.view_size and
                    obj["pos2D"][0] - view_x < self.view_size
                ):
                    obj_pos_sub_matrix[obj["pos2D"][1] - view_y, obj["pos2D"][0] - view_x] = obj_fov_id
                    obj_fov_id += 1
                    objects_in_fov.append(obj)

            # Calcolo bordi robusto
            if view_x < 0:
                start_x = abs(view_x)
                end_x = self.view_size
            elif view_x + self.view_size > flipped_map.shape[1]:
                start_x = 0
                end_x = flipped_map.shape[1] - view_x
            else:
                start_x = 0
                end_x = self.view_size

            if view_y < 0:
                start_y = abs(view_y)
                end_y = self.view_size
            elif view_y + self.view_size > flipped_map.shape[0]:
                start_y = 0
                end_y = flipped_map.shape[0] - view_y
            else:
                start_y = 0
                end_y = self.view_size

            sub_matrix[start_y:end_y, start_x:end_x] += flipped_map[
                view_y + start_y:view_y + end_y,
                view_x + start_x:view_x + end_x
            ]

            rotated_submatrix = np.flip(np.rot90(sub_matrix, k=rot), 0)
            rotated_obj_pos = np.flip(np.rot90(obj_pos_sub_matrix, k=rot), 0)
            rotated_obj_pos[3, 6] = obj_fov_id

            if self.visualization:
                self.ax.imshow(flipped_map, cmap="gray",
                               extent=[0, flipped_map.shape[1], flipped_map.shape[0], 0],
                               vmin=0, vmax=255)
                for obj in self.objects:
                    if obj["detection_steps"] >= self.kalman_init_steps:
                        # Usa i pixel ricalcolati
                        self.ax.scatter(
                            obj["pos2D"][0] + 0.5,
                            obj["pos2D"][1] + 0.5,
                            color=self.IDX_TO_COLOR[obj["color"]],
                            s=100,
                            marker=self.objects_symbols[obj["id"]]
                        )
                self.ax.scatter(agent_x + 0.5, agent_y + 0.5, color="red", marker=marker)
                rect = Rectangle((view_x, view_y), self.view_size, self.view_size,
                                 linewidth=1, edgecolor='green', facecolor='none')
                self.ax.add_patch(rect)
                self.ax.set_xticks(np.arange(0, flipped_map.shape[1] + 1, 1))
                self.ax.set_yticks(np.arange(0, flipped_map.shape[0] + 1, 1))
                self.ax.grid(True, color='black', linestyle='-', linewidth=0.2)
                self.fig.canvas.draw()
                w, h = self.fig.canvas.get_width_height()
                img_numpy = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                self.grid_pub.publish(self.bridge.cv2_to_imgmsg(img_numpy))
                self.map_ready = True
                self.ax.clear()

            # Build observation for policy
            obs = np.zeros((self.view_size, self.view_size, 3))
            obs[rotated_submatrix == 50, 0] = self.real_map_grid_mapping[0]
            obs[rotated_submatrix == 0, 0] = self.real_map_grid_mapping[-1]
            obs[rotated_submatrix == 100, 0] = self.real_map_grid_mapping[100]
            obs[rotated_submatrix == 100, 1] = self.COLOR_TO_IDX["gray"]

            obj_fov_id = 1
            for obj in objects_in_fov:
                obs[rotated_obj_pos == obj_fov_id, 0] = obj["id"]
                obs[rotated_obj_pos == obj_fov_id, 1] = obj["color"]
                obj_fov_id += 1
            
            next_obs = {'image': np.array([obs]), 'mission': (self.mission,)}
            # print(f"next obs {next_obs}")
            print(obs[:, :, 0])
            step = float(self.custom_resolution)

            if self.use_learned_policy and self.map_ready:
                next_obs = preprocess_minigrid_dict_obs_to_tensor(next_obs, self.vocab, max_sentence_length, self.device)
                mask = 1 - (next_obs['text'] == 0).int()
                mask -= (torch.logical_or(next_obs['text'] == 1, next_obs['text'] == 2)).int()
                next_done = torch.zeros(1).to(self.device)

                actions, _, _, _, self.next_lstm_state = self.agent.get_action_and_value(
                    next_obs, self.next_lstm_state, next_done, mask=mask, action_mask=[1, 1, 1, 0, 0, 0, 1]
                )
                
                command = int(actions[0].item())
                print(f"------ Command {command} ------")

                if self.policy_movement(command, cur_yaw_q, cur_x_m, cur_y_m, step) == -1:
                    return
            else:
                if self.init_count < 4:
                    self.get_logger().info(f"Waiting...")
                    time.sleep(3)
                    self.policy_movement(1, cur_yaw_q, cur_x_m, cur_y_m, step)
                    self.init_count += 1
                else:
                    self.use_learned_policy = True

    def reset_agent_memory(self):
        self.next_lstm_state = (
            torch.zeros(self.agent.internal_memory.num_layers, 1, self.agent.internal_memory.lstm_emb_size).to(self.device),
            torch.zeros(self.agent.internal_memory.num_layers, 1, self.agent.internal_memory.lstm_emb_size).to(self.device),
        )

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.is_navigating = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        self.is_navigating = False
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
        elif result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Goal failed/canceled: {result.status}')

def main(args=None):
    rclpy.init(args=args)
    agent_node = AgentNode(tasks=['go to the green ball', 'go to the green box'], use_learned_policy=False)
    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        pass
    finally:
        agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    main()