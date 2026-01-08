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
from obj_agent.two_conv_nets import *
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
#import tf_transformations


class KalmanFilterPositionOnly:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        # Stato iniziale (x, y)
        self.x = np.zeros((2, 1))

        # Covarianza iniziale dell'incertezza
        self.P = np.eye(2)

        # Modello dinamico (oggetto fermo)
        self.F = np.eye(2)

        # Matrice osservazione (misuriamo direttamente posizione)
        self.H = np.eye(2)

        # Rumore del processo (quanto ci fidiamo che l’oggetto sia fermo)
        self.Q = process_variance * np.eye(2)

        # Rumore della misura (quanto sono rumorose le osservazioni)
        self.R = measurement_variance * np.eye(2)

    def predict(self):
        # Lo stato non cambia se l'oggetto è fermo
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z è un vettore colonna con la misura [x, y]
        z = np.reshape(z, (2, 1))

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Aggiornamento dello stato
        y = z - self.H @ self.x
        self.x = self.x + K @ y

        # Aggiornamento della covarianza
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()

class AgentNode(Node):

    def __init__(self, use_learned_policy=False):
        super().__init__('agent_node')

        self.get_logger().info('Initializing agent node...')

        # Group to allow the execution of several callbacks at the same time
        callback_group = ReentrantCallbackGroup()

        # Variable initialization
        self.custom_resolution = 0.43 # meters per cell for the desired resized map
        self.objects = []
        self.locations=[]
        self.obj = []
        self.map = None
        self.resized_map = None
        self.origin = None
        self.robot_pose = [0,0,0]
        self.is_navigating = False
        self.kalman_init_steps = 3
        self.mission = 'go to the green box'

        self.CLASS_TO_IDX = {"ball": 3, "bottle": 2}
        self.COLOR_TO_IDX = {"red": 11, "green": 12, "blue": 13, "purple": 14, "yellow": 15, "gray": 16}
        self.IDX_TO_COLOR = dict(zip(self.COLOR_TO_IDX.values(), self.COLOR_TO_IDX.keys()))
        self.real_map_grid_mapping = {0: 1, # MiniGrid empty
                                 100.: 2, # MiniGrid wall
                                 -1: 0    # MiniGrid empty for now, to clarify
                                 }
        self.objects_symbols = {5: "|",
                           3: "o",
                           2: "s"}
        self.view_size = 7
        # self.direction = 'left'
        self.direction = None
        self.pose_from_topic = None
        
        self.bridge = CvBridge()

        self.device = "cpu"
        self.agent = Agent().to(self.device)
        #print(os.getcwd())
        self.agent.load_state_dict(torch.load("/home/user/ros2_ws/src/pkg/obj_agent/obj_agent/agent_1747644302.cleanrl_model", map_location=self.device), strict=False)
        self.agent.eval()

        self.vocab = Vocabulary(vocab_max_size)
        self.vocab.load_from_json("/home/user/ros2_ws/src/pkg/obj_agent/obj_agent/vocab.json")

        self.next_lstm_state = (
                torch.zeros(self.agent.internal_memory.num_layers, 1, self.agent.internal_memory.lstm_emb_size).to(self.device),
                torch.zeros(self.agent.internal_memory.num_layers, 1, self.agent.internal_memory.lstm_emb_size).to(self.device),
            )

        #qos_profile_image = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=6, reliability=ReliabilityPolicy.RELIABLE)
        qos_profile_image = QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=6, reliability=ReliabilityPolicy.BEST_EFFORT)


        # Map and objects subscriptions
        self.obj_sub = self.create_subscription(DetectedObjectsList,'objects_detected',self.objects_callback,1,callback_group=callback_group)
        self.map_sub = self.create_subscription(OccupancyGrid,'rover/map',self.map_callback,1,callback_group=callback_group)
        self.pose_sub = self.create_subscription(PoseStamped,'rover_tf_pose',self.pose_callback,1,callback_group=callback_group)
        self.grid_pub = self.create_publisher(Image,'grid_map', qos_profile_image)

        self.timer = self.create_timer(1.0, self.agent_call, callback_group=callback_group)

        # Move base client
        self.client = ActionClient(self, NavigateToPose, 'rover/navigate_to_pose')
        # self.client.wait_for_server()

        # Visualization
        self.visualization = True # If true the execution is stopped to visualize each robot step in the map. If false, the robot moves continuously.

        self.get_logger().info('Done!')
        #self.pose_topic = message_filters.Subscriber(self, PoseStamped, '/pose', callback=self.pose_callback)
        
        
        self.use_learned_policy = use_learned_policy  

        if self.visualization:
            self.fig, self.ax = plt.subplots()
            self.fig.set_figwidth(15)
            self.fig.set_figheight(15)

    def set_direction(self):
        self.robot_pose = [*self.pose_from_topic]
        new_value = 2 * np.arctan2(self.robot_pose[2], self.robot_pose[3])
        if new_value>np.radians(135) or new_value<np.radians(-135):
            self.direction = 'right'
        elif new_value>np.radians(-135) and new_value<np.radians(-45):
            self.direction = 'up'
        elif new_value>np.radians(-45) and new_value<np.radians(45):
            self.direction = 'left'
        elif new_value>np.radians(45) and new_value<np.radians(135):
            self.direction = 'down'

    def pose_callback(self, msg):
        #print(f"Pos {msg.pose.pose.position}")
        #print(f"Pos {msg.pose.pose.orientation}")
        self.pose_from_topic = [msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.z, msg.pose.orientation.w]
        self.set_direction()

        #print(self.pose_from_topic)

    def objects_callback(self, msg):
        #self.obj = []
        if self.map:
            width = self.map.info.width
            height = self.map.info.height
            #origin = [self.map.info.origin.position.x,self.map.info.origin.position.y]
            #print(f"objectos {msg.objects}")
            for i, obj in enumerate(msg.objects):
                #print(i)
                obj_ = self.serialize_object(obj)
                current_pos = obj_["pos2D"]
                #print(f"Current pos {current_pos}")
                distances = np.linalg.norm(np.array(current_pos) - np.array([o["real_pos"] for o in self.objects]), axis=1) if self.objects else np.array([])
                if distances.size==0 or not np.any(distances < self.custom_resolution * 3 / 2):
                    #self.obj.append(self.serialize_object(obj))
                    #print(f"First detection {distances}")
                    positions_m = [obj_["pos2D"]]
                    positions_px = self.meters_to_px(positions_m,self.origin,self.resized_map.shape[1])
                    #print(positions_px)
                    #current_pos = [positions_px[i][1],positions_px[i][0]]
                    #print(f"OBJECT COLOR {obj_['trackID']}")
                    #print(f"cls {obj_['cls']}")
                    self.objects.append({"id": self.CLASS_TO_IDX[obj_['cls']], "color": obj_['trackID'], "pos2D": (positions_px[0][1], positions_px[0][0]), 'kf': KalmanFilterPositionOnly(), 'detection_steps': 1, 'real_pos': current_pos})
                    self.objects[-1]['kf'].predict()
                    self.objects[-1]['kf'].update(obj_["pos2D"])
                else:
                    print("Same detection")
                    index = np.argmin(distances)
                    self.objects[index]["detection_steps"] += 1
                    self.objects[index]["kf"].predict()
                    self.objects[index]["kf"].update(obj_["pos2D"])
                    self.objects[index]["real_pos"] = self.objects[index]["kf"].get_state()
                    positions_px = self.meters_to_px([obj_["pos2D"]],self.origin,self.resized_map.shape[1])
                    #print(positions_px)
                    #print(self.objects)
                    self.objects[index]["pos2D"] = (positions_px[0][1], positions_px[0][0])

                
    def map_callback(self, msg):
        self.map = msg
        self.map_processing()   # Aggiorna self.resized_map

    def serialize_object(self, obj):
        corners = [{'x': float(p.x), 'y': float(p.y)} for p in obj.pos2d.points]
        return {
            'cls': str(obj.cls),
            'trackID': int(obj.trackid),
            'pos2D': [obj.pos2d.points[0].x,obj.pos2d.points[0].y]
        }

    def meters_to_px(self, coord_m, ori, h):
        return [
            (
                round((y - ori[1])/self.custom_resolution),
                round(h - ((x - ori[0])/self.custom_resolution))
            )
            for x,y in coord_m
        ]
    
    def map_processing(self):
        if self.map and self.is_navigating is False:
            # Map scaling
            width = self.map.info.width
            height = self.map.info.height
            self.origin = [self.map.info.origin.position.x,self.map.info.origin.position.y]
            #print(f"Origin {origin}")
            occupancy_data = self.map.data
            map_matrix = np.array(occupancy_data).reshape((height,width))
            occ_mask = (map_matrix==100).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
            dilated_mask = cv2.dilate(occ_mask,kernel)
            map_matrix[dilated_mask==1] = 100
            scale_factor = self.map.info.resolution / self.custom_resolution #The desired resolution is self.custom_resolution meters per cell
            new_size = (int(map_matrix.shape[1]*scale_factor),int(map_matrix.shape[0]*scale_factor))
            self.resized_map = cv2.resize(map_matrix, new_size, interpolation = cv2.INTER_NEAREST)

    def agent_call(self):
        print(self.direction)
        if not self.direction:
            return

        if self.map and self.is_navigating is False:
            self.get_logger().info('Calculating step...')
            '''
            # Map scaling
            width = self.map.info.width
            height = self.map.info.height
            origin = [self.map.info.origin.position.x,self.map.info.origin.position.y]
            #print(f"Origin {origin}")
            occupancy_data = self.map.data
            map_matrix = np.array(occupancy_data).reshape((height,width))
            occ_mask = (map_matrix==100).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
            dilated_mask = cv2.dilate(occ_mask,kernel)
            map_matrix[dilated_mask==1] = 100
            scale_factor = self.map.info.resolution / self.custom_resolution #The desired resolution is self.custom_resolution meters per cell
            new_size = (int(map_matrix.shape[1]*scale_factor),int(map_matrix.shape[0]*scale_factor))
            resized_map = cv2.resize(map_matrix, new_size, interpolation = cv2.INTER_NEAREST)
            '''
            # Objects scaling
            # ball: 6, key: 5, box: 7 (from YOLO: sports ball, -, bottle)
            '''
            if self.obj:
                positions_m = [obj['pos2D'] for obj in self.obj]
                positions_px = self.meters_to_px(positions_m,origin,resized_map.shape[1])
                for i, obj in enumerate(self.obj):
                    current_pos = [positions_px[i][1],positions_px[i][0]]
                    distances = np.linalg.norm(np.array(current_pos) - np.array(self.locations), axis=1) if self.locations else np.array([])
                    if distances.size==0 or not np.any(distances<3.0): #Avoiding multiple detections of the same object type in a distance of 3 pixels
                        self.objects.append({"id": self.CLASS_TO_IDX[obj['cls']], "color": obj['trackID'], "pos2D": (positions_px[i][1],positions_px[i][0])})
                        print(f"obj {self.CLASS_TO_IDX[obj['cls']]} pos2D: {obj['pos2D']}")
                        self.locations.append([positions_px[i][1],positions_px[i][0]])
                    else:  
                        min_dist_index = np.argmin(distances)
                        self.objects[i]["pos2D"] = (positions_px[i][1],positions_px[i][0])
            '''

            
            #self.obj = []
            #print(f"robot pose: {self.robot_pose}")
            #robot_px = [round(-origin[1]/self.custom_resolution + self.robot_pose[1]), round((resized_map.shape[1] + origin[0]/self.custom_resolution)-self.robot_pose[0])] 
            robot_px = [round(-self.origin[1]/self.custom_resolution + self.robot_pose[1]/self.custom_resolution), round((self.resized_map.shape[1] + self.origin[0]/self.custom_resolution)-self.robot_pose[0]/self.custom_resolution)] 
            
            #print(robot_px)
            flipped_map = np.flip(self.resized_map, 1)
            flipped_map[flipped_map==0.0] = 50
            flipped_map[flipped_map==-1.0] = 0

            agent_y, agent_x = robot_px

            marker = ""
            rot = 0

            objects_in_fov = []

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

            obj_fov_id = 1
            for obj in self.objects:
                if obj["detection_steps"] >= self.kalman_init_steps and obj["pos2D"][1] - view_y >= 0 and obj["pos2D"][0] - view_x >= 0 and obj["pos2D"][1] - view_y < self.view_size and obj["pos2D"][0] - view_x < self.view_size:
                    obj_pos_sub_matrix[obj["pos2D"][1] - view_y, obj["pos2D"][0] - view_x] = obj_fov_id
                    obj_fov_id += 1
                    objects_in_fov.append(obj)

            start_x = np.abs(view_x) if view_x < 0 else 0
            end_x = self.view_size - np.abs(view_x) if view_x + self.view_size > flipped_map.shape[1] else self.view_size
            start_y = np.abs(view_y) if view_y < 0 else 0
            end_y = self.view_size - np.abs(view_y) if view_y + self.view_size > flipped_map.shape[0] else self.view_size
            
            #print(start_y, end_y, start_x, end_x, view_x, view_y)
            #print(sub_matrix[start_y:end_y, start_x:end_x].shape)
            #print(flipped_map[view_y+start_y:view_y+end_y, view_x+start_x:view_x+end_x].shape)

            sub_matrix[start_y:end_y, start_x:end_x] += flipped_map[view_y+start_y:view_y+end_y, view_x+start_x:view_x+end_x]
            rotated_submatrix = np.flip(np.rot90(sub_matrix, k=rot), 0)
            rotated_obj_pos = np.flip(np.rot90(obj_pos_sub_matrix, k=rot), 0)
            rotated_obj_pos[3, 6] = obj_fov_id
            #print(sub_matrix)
            #print(rotated_obj_pos)
            
            if self.visualization:

                self.ax.imshow(flipped_map, cmap="gray", extent=[0, flipped_map.shape[1], flipped_map.shape[0], 0], vmin=0, vmax=255)
                for obj in self.objects:
                    if obj["detection_steps"] >= self.kalman_init_steps:
                        self.ax.scatter(obj["pos2D"][0] + 0.5, obj["pos2D"][1] + 0.5, color=self.IDX_TO_COLOR[obj["color"]], s=100, marker=self.objects_symbols[obj["id"]])
                    
                self.ax.scatter(agent_x+0.5, agent_y+0.5, color="red", marker=marker)
                rect = Rectangle((view_x, view_y), self.view_size, self.view_size, linewidth=1, edgecolor='green', facecolor='none')
                self.ax.add_patch(rect)
                self.ax.set_xticks(np.arange(0, flipped_map.shape[1] + 1, 1))
                self.ax.set_yticks(np.arange(0, flipped_map.shape[0] + 1, 1))
                self.ax.grid(True, color='black', linestyle='-', linewidth=0.2)
                self.fig.canvas.draw()
                w, h = self.fig.canvas.get_width_height()
                img_numpy = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                #img = ax.imshow(flipped_map, cmap="gray", extent=[0, flipped_map.shape[1], flipped_map.shape[0], 0], vmin=0, vmax=255)
                #img_numpy = img.get_array()
                #print(img_numpy.shape)
                print("\n----------\nvisualization\n---------\n")
                self.grid_pub.publish(self.bridge.cv2_to_imgmsg(img_numpy))    
                #plt.clf()
                # plt.cla()
                self.ax.clear()
                #plt.close()

                #fig2, ax2 = plt.subplots()
                #ax2.imshow(rotated_submatrix, cmap="gray", extent=[0, self.view_size, self.view_size, 0], vmin=0, vmax=255)
            
            '''
            obj_fov_id = 1
            for obj in objects_in_fov:
                w = np.where(rotated_obj_pos == obj_fov_id)
                if w[0].squeeze().size > 0:
                    if self.visualization:
                        ax2.scatter(w[1].squeeze() + 0.5, w[0].squeeze() + 0.5, color=self.IDX_TO_COLOR[obj["color"]], s=400, marker=self.objects_symbols[obj["id"]])
                    else:
                        continue
                obj_fov_id += 1
            '''
            w = np.where(rotated_obj_pos == obj_fov_id)

            #if self.visualization:
            #    ax2.scatter(w[1].squeeze() + 0.5, w[0].squeeze() + 0.5, color="red", marker="<", s=400)     # Agent
            #    ax2.grid(True, color='black', linestyle='-', linewidth=0.2)
            #    plt.show()

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

            next_obs = {'image': np.array([obs]),
                        'mission': (self.mission,)}
            next_obs = preprocess_minigrid_dict_obs_to_tensor(next_obs, self.vocab, max_sentence_length, self.device)
            mask = 1 - (next_obs['text'] == 0).int()
            mask -= (torch.logical_or(next_obs['text'] == 1, next_obs['text'] == 2)).int()
            next_done = torch.zeros(1).to(self.device)

            actions, _, _, _, self.next_lstm_state, _ = self.agent.get_action_and_value(next_obs, self.next_lstm_state, next_done, mask=mask) 

            command = actions[0]
            print(f"------ Command {command} ------") # 6: done
            #print(self.robot_pose)
            
            if self.use_learned_policy:
                print(f"rot {rot}")
                if command==2:
                    self.get_logger().info('Moving forward...')
                    if rot==0:
                        self.robot_pose[1] = self.robot_pose[1] + 1 * self.custom_resolution
                    elif rot==-1:
                        self.robot_pose[0] = self.robot_pose[0] + 1 * self.custom_resolution
                    elif rot==1:
                        self.robot_pose[0] = self.robot_pose[0] - 1 * self.custom_resolution
                    else: # -90 degrees
                        self.robot_pose[1] = self.robot_pose[1] - 1 * self.custom_resolution
                elif command==1:
                    self.get_logger().info('Turning right...')
                    if rot==0:
                        self.direction = 'up'
                        self.robot_pose[2] = -90
                        rot = 1
                    elif rot==-1:
                        self.direction = 'left'
                        self.robot_pose[2] = 0
                        rot = 0
                    elif rot==2:
                        self.direction = 'down'
                        self.robot_pose[2] = 90
                        rot = -1
                    else: # -90 degrees
                        self.direction = 'right'
                        self.robot_pose[2] = 180
                        rot = 2
                elif command==0:
                    self.get_logger().info('Turning left...')
                    if rot==0:
                        self.direction = 'down'
                        self.robot_pose[2] = 90
                        rot = -1
                    elif rot==-1:
                        self.direction = 'right'
                        self.robot_pose[2] = 180
                        rot = 2
                    elif rot==2:
                        self.direction = 'up'
                        self.robot_pose[2] = -90
                        rot = 1
                    else: # -90 degrees
                        self.direction = 'left'
                        self.robot_pose[2] = 0
                        rot = 0

                print(f"GOAL MSG BEFORE {self.robot_pose[1]}, {self.robot_pose[0]}")
                # Send goal to robot
                goal_msg = NavigateToPose.Goal()
                goal_msg.pose.header.frame_id = 'map'
                goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

                goal_msg.pose.pose.position.x = self.robot_pose[1]#*self.custom_resolution
                goal_msg.pose.pose.position.y = self.robot_pose[0]#*self.custom_resolution


                if rot==0:
                    orientation=[0.0,1.0]
                elif rot==-1:
                    orientation=[0.707,0.707]
                elif rot==2:
                    orientation=[1.0,0.0]
                else: # -90 degrees
                    orientation=[-0.707,0.707]
                goal_msg.pose.pose.orientation.z = orientation[0]
                goal_msg.pose.pose.orientation.w = orientation[1]

                if command==3 or command==6:
                    self.get_logger().info('Object found!')
                else:
                    self.get_logger().info('Moving...')
                    self.is_navigating = True
                    send_goal_future = self.client.send_goal_async(goal_msg)
                    send_goal_future.add_done_callback(self.goal_response_callback)
            else:
                #print(f"Pose from topic {self.pose_from_topic}")
                if  self.pose_from_topic is not None:
                    self.set_direction()





    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted, waiting for result...')

        # Wait for result asynchronously
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()

        self.is_navigating = False

        # Check the result status using GoalStatus
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
        elif result.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().info('Goal aborted!')
        elif result.status == GoalStatus.STATUS_REJECTED:
            self.get_logger().info('Goal rejected!')
        elif result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Goal canceled!')
        else:
            self.get_logger().info(f'Goal failed with status: {result.status}')

def main(args=None):
    rclpy.init(args=args)
    agent_node = AgentNode(use_learned_policy=True)

    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        pass
    finally:
        agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()