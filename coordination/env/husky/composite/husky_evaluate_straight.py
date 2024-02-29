from collections import OrderedDict
import numpy as np
import math
from env.husky.husky import HuskyEnv
from env.transform_utils import up_vector_from_quat, Y_vector_from_quat, \
    l2_dist, cos_dist, sample_quat, X_vector_from_quat, alignment_heading_difference, \
    Y_vector_overlapping, movement_heading_difference, calculate_rotation_direction, \
    get_quaternion_to_next_cpt




def path_planning(box_pos, box_quat, goal_pos, goal_quat):
    # should I use hybrid a*? or RTT? or something else?
    raise ValueError



class HuskyEvaluateStraightEnv(HuskyEnv):
    def __init__(self, **kwargs):
        self.name = 'husky_evaluate_straight'
        super().__init__('husky_evaluate_straight.xml', **kwargs)

        # Env info
        self.ob_shape = OrderedDict([("husky_1", 15), ("husky_2", 15),
                                     ("box_1", 12), ("box_2", 12),
                                     ("relative_info", 10)])
        
        
        self.action_space.decompose(OrderedDict([("husky_1", 2), ("husky_2", 2)]))

        # Env config
        self._env_config.update({
            'random_husky_pos': 0.01,
            'random_goal_pos': 0.01,
            #'random_goal_pos': 0.5,
            'dist_threshold': 0.1,
            'loose_dist_threshold': 0.8,
            'goal_box_cos_dist_coeff_threshold': 0.9,
            'box_linear_vel_reward': 500,
            'dist_reward': 10,
            'alignment_reward': 30,
            'goal_dist_reward': 30,
            'goal1_dist_reward': 10,
            'goal2_dist_reward': 10,
            'move_heading_reward': 30,
            'linear_vel_reward': 50,
            'success_reward': 500,
            'bonus_reward': 20,

            'quat_reward': 30, 
            'alive_reward': 30,
            'die_penalty': 500,
            'sparse_reward': 0,
            'init_randomness': 0.01,
            #'max_episode_steps': 400,
            'max_episode_steps': 400,
        }) 
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        self._husky1_push = False
        self._husky2_push = False
        self._success_count = 0
        self._experiment_type = None


    def _vector_field_rad(self, robot_control_pos, robot_control_forward_vec, robot_rotate_pos, target_pos, robot_rotate_box_dist, boxes_dist):
        k = 0.5     # the larger k is, the more abrupt transition of the field

        #suggested_dist = (robot_rotate_box_dist + boxes_dist) / 2
        suggested_dist = boxes_dist

        dist_rr_t = (l2_dist(robot_control_pos, robot_rotate_pos) - suggested_dist) * k
        gamma_rc_rr = np.arctan2((robot_control_pos[1] - robot_rotate_pos[1]), (robot_control_pos[0] - robot_rotate_pos[0]))
        sign = calculate_rotation_direction(robot_control_pos, robot_control_forward_vec, target_pos) 

        desire_rad = gamma_rc_rr + (sign * np.pi/2) + (sign * np.arctan(dist_rr_t))
        cur_rad = -np.arctan2(robot_control_forward_vec[1], robot_control_forward_vec[0])

        if(desire_rad > np.pi):
            desire_rad = -(2 * np.pi - desire_rad)
        if(desire_rad < -np.pi):
            desire_rad = (2 * np.pi + desire_rad)

        return desire_rad, cur_rad


    def _step(self, a):

        nonhierarchical_with_spatial_single = False   # no hierarchy, with relative spatial info, single policy
        nonhierarchical_nospatial_single = False  # no hierarchy, no relative spatial info, single policy
        nonhierarchical_with_spatial_multi = False  # non-hierarchical, with relative spatial info, multi policies
        nonhierarchical_nospatial_multi = False  # non-hierarchical, without relative spatial info, multi policies 

        hierarchical_Uniform_Vector = False           # set this to True will evaluate the reward based on our uniform vector field approach
        hierarchical_NoSpatial_Baseline = True     # set this to True will evaluate the reward without relative spatial info

        if hierarchical_NoSpatial_Baseline:
            self._experiment_type = "Hierarchical without relative-spatial info" 
        elif hierarchical_Uniform_Vector:
            self._experiment_type = "Hierarchical Uniform VectorField"
        elif nonhierarchical_with_spatial_single:
            self._experiment_type = "Non-hierarchical with relative spatial info, single policy" 
        elif nonhierarchical_nospatial_single:
            self._experiment_type = "Non-hierarchical without relative spatial info, single policy"
        elif nonhierarchical_nospatial_multi:
            self._experiment_type = "Non-hierarchical without relative spatial info, multi policies"
        elif nonhierarchical_with_spatial_multi:
            self._experiment_type = "Non-hierarchical with relative spatial info, multi policies"
        else:
            self._experiment_type = "Hierarchical with relative spatial info"

        husky1_pos_before = self._get_pos('husky_1_geom')
        husky2_pos_before = self._get_pos('husky_2_geom')

        box1_pos_before = self._get_pos('box_geom1')
        box2_pos_before = self._get_pos('box_geom2')
        box_pos_before = self._get_pos('box')
        box_quat_before = self._get_quat('box')


        self.do_simulation(a)


        husky1_pos = self._get_pos('husky_1_geom')
        husky2_pos = self._get_pos('husky_2_geom')
        box1_pos = self._get_pos('box_geom1')
        box2_pos = self._get_pos('box_geom2')
        box_pos = self._get_pos('box')
        box_quat = self._get_quat('box')

        final_goal_pos = self._get_pos('goal')      # the final goal
        final_goal_quat = self._get_quat('goal')

        goal1_pos = self._get_pos('waypoint_1')     # waypoint 1
        goal2_pos = self._get_pos('waypoint_2')     # waypoint 2
        goal_pos = self._get_pos('waypoint')        # overall waypoint
        goal_quat = self._get_quat('waypoint')      



        #TODO
        planned_trajectory = path_planning(box_pos, box_quat, final_goal_pos, final_goal_quat)
        



        ob = self._get_obs()
        done = False
        ctrl_reward = self._ctrl_reward(a)

        '''
        goal_forward = forward_vector_from_quat(goal_quat)
        box_forward = forward_vector_from_quat(box_quat)
        box_forward_before = forward_vector_from_quat(box_quat_before)
        '''

        #goal_forward_before = right_vector_from_quat(goal_quat_before)    
        goal_forward = X_vector_from_quat(goal_quat)
        box_forward = X_vector_from_quat(box_quat)
        box_forward_before = X_vector_from_quat(box_quat_before)

        # goal 1
        goal1_dist = l2_dist(goal1_pos, box1_pos)
        goal1_dist_before = l2_dist(goal1_pos, box1_pos_before)

        # goal 2
        goal2_dist = l2_dist(goal2_pos, box2_pos)
        goal2_dist_before = l2_dist(goal2_pos, box2_pos_before)

        husky1_quat = self._get_quat('husky_robot_1')
        husky2_quat = self._get_quat('husky_robot_2')


        '''
            Reward & Penalty
            PART 1, 2, 3: control the formation of huskys
            PART 4, 5: distance between husky and box, box and goal
            PART 6: move heading of huskys
        ''' 
        # PART 1: Forward parallel between two Huskys (checking forward vector)
        # X_vector_from_quat is same as right_vector_from_quat, which is the actual forward vector of robot
        husky1_forward_vec = X_vector_from_quat(husky1_quat)
        husky2_forward_vec = X_vector_from_quat(husky2_quat)
        huskys_forward_align_coeff, dir = alignment_heading_difference(husky1_forward_vec, husky2_forward_vec)
        #huskys_forward_align_reward = self._env_config["alignment_reward"] * np.exp(huskys_forward_align_coeff)
        huskys_forward_align_reward = huskys_forward_align_coeff * self._env_config["alignment_reward"]


        # PART 2: Right vector overlapping between two Huskys (checking if two vectors are on the same line and same direction)
        # if Part 2 is gauranteed, then Part 1 is gauranteed 
        husky1_right_vec = Y_vector_from_quat(husky1_quat)
        husky2_right_vec = Y_vector_from_quat(husky2_quat)
        huskys_right_align_coeff = Y_vector_overlapping(husky1_right_vec, husky2_right_vec, husky1_pos, husky2_pos)
        huskys_right_align_reward = huskys_right_align_coeff * self._env_config["alignment_reward"]
        #huskys_right_align_reward = self._env_config["alignment_reward"] * np.exp(huskys_right_align_coeff)


        # PART 3: Distance between two Huskys (to avoid huskys collision)
        boxes_dist = l2_dist(box1_pos, box2_pos)        # desired distance
        huskys_dist = l2_dist(husky1_pos, husky2_pos)
        
        diff = abs(boxes_dist - huskys_dist)
        huskys_dist_reward = -self._env_config["dist_reward"] * (diff ** 2) * 10


        # PART 4: Linear distance between one husky and one box (Husky - Box)
        husky1_box_dist = l2_dist(husky1_pos, box1_pos)     # husky1 to box1
        husky2_box_dist = l2_dist(husky2_pos, box2_pos)     # husky2 to box2
        #husky1_box_dist_reward = -husky1_box_dist * self._env_config["dist_reward"]
        #husky2_box_dist_reward = -husky2_box_dist * self._env_config["dist_reward"]
        #husky1_box_dist_reward = (5 - husky1_box_dist) * self._env_config["dist_reward"]
        #husky2_box_dist_reward = (5 - husky2_box_dist) * self._env_config["dist_reward"]

        '''
            vector field based reward shaping
        '''
        husky1_box_dist_reward = 2 / husky1_box_dist * self._env_config["dist_reward"]
        husky2_box_dist_reward = 2 / husky2_box_dist * self._env_config["dist_reward"]
        huskys_box_dist_reward = husky1_box_dist_reward + husky2_box_dist_reward
        
        # PART 5: Linear distance between box and goal
        goal1_box_dist_before = l2_dist(goal1_pos, box1_pos_before)
        goal2_box_dist_before = l2_dist(goal2_pos, box2_pos_before)
        goal1_box_dist = l2_dist(goal1_pos, box1_pos)
        goal2_box_dist = l2_dist(goal2_pos, box2_pos)
        goal1_box_dist_diff = goal1_box_dist_before - goal1_box_dist
        goal2_box_dist_diff = goal2_box_dist_before - goal2_box_dist
        #goal1_box_dist_reward = -goal1_box_dist * self._env_config["dist_reward"]
        #goal2_box_dist_reward = -goal2_box_dist * self._env_config["dist_reward"]
        #goal1_box_dist_reward = (1 - goal1_box_dist) * self._env_config["goal_dist_reward"]
        #goal2_box_dist_reward = (1 - goal2_box_dist) * self._env_config["goal_dist_reward"]


        
        goal1_box_dist_reward = 4 / goal1_box_dist * self._env_config["dist_reward"]
        goal2_box_dist_reward = 4 / goal2_box_dist * self._env_config["dist_reward"]

        goal_box_dist_reward = goal1_box_dist_reward + goal2_box_dist_reward

        # PART 6: Movement heading of husky to box
        husky1_move_coeff = movement_heading_difference(box1_pos, husky1_pos, husky1_forward_vec)
        husky2_move_coeff = movement_heading_difference(box2_pos, husky2_pos, husky2_forward_vec)
        #husky1_move_heading_reward = husky1_move_coeff * self._env_config["move_heading_reward"]
        #husky2_move_heading_reward = husky2_move_coeff * self._env_config["move_heading_reward"]
        #huskys_move_heading_reward = husky1_move_heading_reward + husky2_move_heading_reward

        # PART 7: Box velocity
        #box1_linear_vel =  l2_dist(box1_pos, box1_pos_before)
        #box2_linear_vel = l2_dist(box2_pos, box2_pos_before)
        #box1_linear_vel_reward = -self._env_config["linear_vel_reward"] * (((goal1_box_dist - box1_linear_vel) ** 2) / 10)  # square make the final reward too big
        #box2_linear_vel_reward = -self._env_config["linear_vel_reward"] * (((goal2_box_dist - box2_linear_vel) ** 2) / 10)

        #box1_linear_vel_reward = box1_linear_vel * self._env_config["box_linear_vel_reward"]
        #box2_linear_vel_reward = box2_linear_vel * self._env_config["box_linear_vel_reward"]

        box1_forward_vec = X_vector_from_quat(box_quat)
        box2_forward_vec = box1_forward_vec
        #box1_move_direction = np.sign(goal1_box_dist_diff)
        #box2_move_direction = np.sign(goal2_box_dist_diff)

        box1_move_coeff = movement_heading_difference(goal1_pos, box1_pos, box1_forward_vec)
        box2_move_coeff = movement_heading_difference(goal2_pos, box2_pos, box2_forward_vec)
        box1_linear_vel_reward = goal1_box_dist_diff * self._env_config["box_linear_vel_reward"]
        box2_linear_vel_reward = goal2_box_dist_diff * self._env_config["box_linear_vel_reward"]

        sub_boxes_linear_vel_reward = (box1_linear_vel_reward * box1_move_coeff) + (box2_linear_vel_reward * box2_move_coeff)
        

        # PART 8: Cos distance between box and goal
        goal_box_cos_dist_coeff = 1 - cos_dist(goal_forward, box_forward) 
        goal_box_cos_dist_reward = goal_box_cos_dist_coeff * self._env_config["quat_reward"]


        # PART 9: vector fiel        
        # NOTE: this is a experiment feature!
        husky1_desire_rad, husky1_cur_rad = self._vector_field_rad(husky1_pos, husky1_forward_vec, husky2_pos, box1_pos, l2_dist(husky2_pos, box1_pos), boxes_dist)
        husky2_desire_rad, husky2_cur_rad = self._vector_field_rad(husky2_pos, husky2_forward_vec, husky1_pos, box2_pos, l2_dist(husky1_pos, box2_pos), boxes_dist) 
        
        husky1_rad_diff = abs(husky1_desire_rad - husky1_cur_rad)
        husky2_rad_diff = abs(husky2_desire_rad - husky2_cur_rad)

        husky1_rad_reward = -husky1_rad_diff * 30
        husky2_rad_reward = -husky2_rad_diff * 30
        huskys_rad_reward = husky1_rad_reward + husky2_rad_reward

        reward = 0

        '''
            Failure Check
        '''
        if huskys_dist < (boxes_dist * 0.8)  or huskys_dist > boxes_dist * 1.5:   # huskys are too close or too far away 
            done = True
        if husky1_box_dist > 4.0 or husky2_box_dist > 4.0: # husky is too far away from box 
            done = True
        if goal1_box_dist > 4.5 and goal2_box_dist > 4.5:
            done = True
        die_penalty = -self._env_config["die_penalty"] if done else 0
        if done:    # something fail
            reward += die_penalty


        # give some bonus if pass all failure check
        if done != True:
            reward += self._env_config["alive_reward"]


        '''
            Success Check
        '''
        success_reward = 0
        '''
        if (goal1_dist <= self._env_config["dist_threshold"] and \
            goal2_dist <= self._env_config["dist_threshold"]) or \
            (goal1_dist <= self._env_config["loose_dist_threshold"] and \
             goal2_dist <= self._env_config["loose_dist_threshold"] and \
             goal_box_cos_dist_coeff >= self._env_config["goal_box_cos_dist_coeff_threshold"]):
        '''


        if (goal1_dist <= self._env_config["loose_dist_threshold"] and \
             goal2_dist <= self._env_config["loose_dist_threshold"] and \
             goal_box_cos_dist_coeff > self._env_config["goal_box_cos_dist_coeff_threshold"]):
                # if both goal1 and goal2 suffice, then overall goal should suffice
                #and goal_quat < self._env_config["quat_threshold"]:
            self._success_count += 1
            success_reward = 10
            reward += success_reward
            if self._success_count >= 1:
                self._success = True
                success_reward = self._env_config["success_reward"]
                reward += success_reward
                done = True

        reward += ctrl_reward




        '''
            Update waypoint
        '''


        goal_wpt_dist = l2_dist(final_goal_pos, goal_pos)       # final goal - waypoint distance
        goal_box_dist = l2_dist(final_goal_pos, box_pos)        # final goal - box distance
        box_wpt_dist = l2_dist(goal_pos, box_pos)               # waypoint - box distance

        wpt_pos = goal_pos
        wpt_x, wpt_y, wpt_next_x, wpt_next_y = 0, 0, 0, 0

        quaternion = np.zeros(4)

        '''
        if(goal_wpt_dist > goal_box_dist):
            planned_trajectory = path_planning(box_pos, box_quat, final_goal_pos, final_goal_quat)
        
        if(box_wpt_dist > 2.5):     # Box is too far away from the robots, which indicates a drifting
            drift_away = True
        '''
            

        #if (len(planned_trajectory) > 1 and (box_wpt_dist < 1.8)):
        while (len(planned_trajectory) > 1) and box_wpt_dist < 1.8  :
            # update a waypoint, once
            (wpt_x, wpt_y) = planned_trajectory.pop(0)
            (wpt_next_x, wpt_next_y) = planned_trajectory[0]

            wpt_pos = np.array([wpt_x, wpt_y, 0.3]) 
            box_pos = self._get_pos('box')

            box_wpt_dist = l2_dist(wpt_pos, box_pos)

            wpt_next_pos = np.array([wpt_next_x, wpt_next_y, 0])
            
            quaternion = get_quaternion_to_next_cpt(box_pos, wpt_next_pos)

            self._set_pos('cpt_1', np.array([wpt_x, wpt_y, 0.3]))
            self._set_quat('cpt_1', quaternion)


            '''
            while(box_wpt_dist < 1.8):                
                if(len(planned_trajectory) > 1):
                    (rx,ry) = checkpoints.pop(0)
                    (nrx, nry) = checkpoints[0]
                    cpt_pos = np.array([rx, ry, 0.3])
                    box_pos = self._get_pos('box')

                    cpt_box_dist = l2_dist(cpt_pos, box_pos)
                    goal_cpt_dist = l2_dist(final_pos, np.array([rx, ry, 0.3]))

                    pos_1 = np.array([rx, ry, 0])
                    pos_2 = np.array([nrx, nry, 0])

                    quaternion = get_quaternion_to_next_cpt(box_pos, pos_2)

                    self._set_pos('cpt_1', np.array([rx, ry, 0.3]))     # update cpt pos
                    self._set_quat('cpt_1', quaternion)   
                            
                else:
                    break
            '''


        goal1_dist_reward = 0
        goal2_dist_reward = 0
        box_goal_linear_vel_reward = 0
        huskys_box_linear_vel_reward = 0

        if self._env_config['sparse_reward']:
            self._reward = reward = self._success == 1
        else:
            if(hierarchical_NoSpatial_Baseline == True):      # No relative spatial info used for reward function
                goal1_dist_reward = self._env_config["goal1_dist_reward"] if goal1_dist < self._env_config["loose_dist_threshold"] else 0
                goal2_dist_reward = self._env_config["goal2_dist_reward"] if goal2_dist < self._env_config["loose_dist_threshold"] else 0
                box_goal_linear_vel_reward = (l2_dist(box_pos_before, goal_pos) - l2_dist(box_pos, goal_pos)) * self._env_config["box_linear_vel_reward"] + goal1_dist_reward + goal2_dist_reward

                huskys_box_linear_vel_reward = (l2_dist(husky1_pos_before, box1_pos_before) - l2_dist(husky1_pos, box1_pos)) * self._env_config["dist_reward"] + \
                                           (l2_dist(husky2_pos_before, box2_pos_before) - l2_dist(husky2_pos, box2_pos)) * self._env_config["dist_reward"] 

                huskys_box_linear_vel_reward *= 10

                reward = reward + huskys_box_linear_vel_reward + box_goal_linear_vel_reward + goal_box_cos_dist_reward


            elif(hierarchical_Uniform_Vector == True):
                reward = reward + huskys_rad_reward + huskys_box_dist_reward + goal_box_dist_reward + sub_boxes_linear_vel_reward + goal_box_cos_dist_reward + huskys_dist_reward



            elif(nonhierarchical_nospatial_single == True or nonhierarchical_nospatial_multi == True):
                goal1_dist_reward = self._env_config["goal1_dist_reward"] if goal1_dist < self._env_config["loose_dist_threshold"] else 0
                goal2_dist_reward = self._env_config["goal2_dist_reward"] if goal2_dist < self._env_config["loose_dist_threshold"] else 0
                box_goal_linear_vel_reward = (l2_dist(box_pos_before, goal_pos) - l2_dist(box_pos, goal_pos)) * self._env_config["box_linear_vel_reward"] + goal1_dist_reward + goal2_dist_reward

                huskys_box_linear_vel_reward = (l2_dist(husky1_pos_before, box1_pos_before) - l2_dist(husky1_pos, box1_pos)) * self._env_config["dist_reward"] + \
                                           (l2_dist(husky2_pos_before, box2_pos_before) - l2_dist(husky2_pos, box2_pos)) * self._env_config["dist_reward"] 
                
                huskys_box_linear_vel_reward *= 10

                reward = reward + huskys_box_linear_vel_reward + box_goal_linear_vel_reward + goal_box_cos_dist_reward

            elif(nonhierarchical_with_spatial_single == True or nonhierarchical_with_spatial_multi == True):    # NOTE: xxx_dist_reward focus on rewarding based on current distance, xx_linear_reward focus on rewarding the distance has passed through (which is velocity)
                huskys_box_linear_vel_reward = (l2_dist(husky1_pos_before, box1_pos_before) - l2_dist(husky1_pos, box1_pos)) * self._env_config["dist_reward"] * husky1_move_coeff+ \
                                           (l2_dist(husky2_pos_before, box2_pos_before) - l2_dist(husky2_pos, box2_pos)) * self._env_config["dist_reward"] * husky2_move_coeff

                goal1_dist_reward = self._env_config["goal1_dist_reward"] if goal1_dist < self._env_config["loose_dist_threshold"] else 0
                goal2_dist_reward = self._env_config["goal2_dist_reward"] if goal2_dist < self._env_config["loose_dist_threshold"] else 0

                # count the whole box velocity toward the goal
                box_goal_linear_vel_reward = (l2_dist(box_pos_before, goal_pos) - l2_dist(box_pos, goal_pos)) * self._env_config["box_linear_vel_reward"] + goal1_dist_reward + goal2_dist_reward

                
                huskys_box_linear_vel_reward *= 10

                reward = reward \
                    + huskys_box_linear_vel_reward \
                    + huskys_dist_reward \
                    + huskys_box_dist_reward \
                    + goal_box_dist_reward \
                    + huskys_right_align_reward \
                    + sub_boxes_linear_vel_reward \
                    + goal_box_cos_dist_reward \
                    + box_goal_linear_vel_reward
                 
            else:       # hierarchical with spatial info   # NOTE: xxx_dist_reward focus on rewarding based on current distance, xx_linear_reward focus on rewarding the distance has passed through (which is velocity)
                huskys_box_linear_vel_reward = (l2_dist(husky1_pos_before, box1_pos_before) - l2_dist(husky1_pos, box1_pos)) * self._env_config["dist_reward"] * husky1_move_coeff+ \
                                           (l2_dist(husky2_pos_before, box2_pos_before) - l2_dist(husky2_pos, box2_pos)) * self._env_config["dist_reward"] * husky2_move_coeff

                goal1_dist_reward = self._env_config["goal1_dist_reward"] if goal1_dist < self._env_config["loose_dist_threshold"] else 0
                goal2_dist_reward = self._env_config["goal2_dist_reward"] if goal2_dist < self._env_config["loose_dist_threshold"] else 0

                # count the whole box velocity toward the goal, move coefficient is not considered
                box_goal_linear_vel_reward = (l2_dist(box_pos_before, goal_pos) - l2_dist(box_pos, goal_pos)) * self._env_config["box_linear_vel_reward"] + goal1_dist_reward + goal2_dist_reward

                
                huskys_box_linear_vel_reward *= 10

                reward = reward \
                    + huskys_box_linear_vel_reward \
                    + huskys_dist_reward \
                    + huskys_box_dist_reward \
                    + goal_box_dist_reward \
                    + huskys_right_align_reward \
                    + sub_boxes_linear_vel_reward \
                    + goal_box_cos_dist_reward \
                    + box_goal_linear_vel_reward

                    #+ huskys_rad_reward
                    #+ huskys_linear_vel_reward
                    #+ huskys_move_heading_reward \
            self._reward = reward


        info = {"Experiment type": self._experiment_type,
                "success": self._success,
                "Total reward": self._reward,
                "huskys right align reward": huskys_right_align_reward,
                "husky-to-husky dist reward": huskys_dist_reward,
                "husky-to-box dist reward": huskys_box_dist_reward,
                "goal-box dist reward": goal_box_dist_reward,
                "sub-boxes velocity reward": sub_boxes_linear_vel_reward,
                "goal-to-box cos dist reward": goal_box_cos_dist_reward,
                "---------- New Add-on reward ----------": 0,
                "goal1 dist reward": goal1_dist_reward,
                "goal2 dist reward": goal2_dist_reward,
                "box-goal linear vel reward": box_goal_linear_vel_reward,
                "huskys-box linear vel reward": huskys_box_linear_vel_reward,
                "---------- Uniform Vector Field ----------": 0,
                "huskys rad reward": huskys_rad_reward,
                "husky1 desire rad": husky1_desire_rad,
                "husky1 cur rad": husky1_cur_rad,
                "husky1 rad diff": husky1_rad_diff,
                "reward: husky1 rad reward": husky1_rad_reward,
                "husky2 desire rad": husky2_desire_rad,
                "husky2 cur rad": husky2_cur_rad,
                "husky2 rad diff": husky2_rad_diff,
                "reward: husky2 rad reward": husky2_rad_reward,
                "----------": 0,
                "coeff: huskys right align coeff": huskys_right_align_coeff,
                "coeff: husky1 move heading coeff": husky1_move_coeff,
                "coeff: husky2 move heading coeff": husky2_move_coeff,
                "coeff: box_goal_cos_dist": goal_box_cos_dist_coeff,
                "die_penalty": die_penalty,
                "reward_success": success_reward,
                "huskys_dist": huskys_dist,
                "box1_ob": ob['box_1'],
                "box2_ob": ob['box_2'],
                }

        return ob, reward, done, info

    def _get_obs(self):
        # husky
        qpos = self.data.qpos
        qvel = self.data.qvel

        husky1_pos = self._get_pos('husky_robot_1')
        husky2_pos = self._get_pos('husky_robot_2')
        husky1_quat = self._get_quat('husky_robot_1') 
        husky2_quat = self._get_quat('husky_robot_2')
        husky1_vel = self._get_vel('husky_1_body')
        husky2_vel = self._get_vel('husky_2_body')

        husky1_forward_vec = X_vector_from_quat(husky1_quat)
        husky2_forward_vec = X_vector_from_quat(husky2_quat)

        husky1_right_vec = Y_vector_from_quat(husky1_quat)
        husky2_right_vec = Y_vector_from_quat(husky2_quat)
        

        # box
        box_quat = self._get_quat('box') 
        box_forward = X_vector_from_quat(box_quat)

        box1_pos = self._get_pos('box_geom1')
        box2_pos = self._get_pos('box_geom2')

        #box1_quat1 = self._get_quat('box_geom1')
        #box2_quat2 = self._get_quat('box_geom2')
    
        box1_forward = self._get_right_vector('box_geom1')
        box2_forward = self._get_right_vector('box_geom2')
        box_vel = self._get_vel("box_body")

        # goal --> waypoint
        # NOTE: the goal here is actually the info of waypoint
        goal_quat = self._get_quat('cpt_1')
        goal_forward = X_vector_from_quat(goal_quat)
        goal1_pos = self._get_pos('cpt_1_geom1')
        goal2_pos = self._get_pos('cpt_1_geom2')

        # check nan value, not sure what cause this issue
        has_nan = any(math.isnan(value) for value in goal1_pos) or any(math.isnan(value) for value in goal2_pos) 
        if has_nan:
            goal1_pos = self._get_pos('cpt_1')
            goal2_pos = self._get_pos('cpt_1')


        # relative info
        husky1_move_coeff = movement_heading_difference(box1_pos, husky1_pos, husky1_forward_vec, "forward") 
        husky2_move_coeff = movement_heading_difference(box2_pos, husky2_pos, husky2_forward_vec, "forward")
        huskys_dist = l2_dist(husky1_pos, husky2_pos)
        huskys_forward_align_coeff, dir = alignment_heading_difference(husky1_forward_vec, husky2_forward_vec)
        huskys_right_align_coeff = Y_vector_overlapping(husky1_right_vec, husky2_right_vec, husky1_pos, husky2_pos)

        goal_box_cos_dist_coeff = 1 - cos_dist(box_forward, goal_forward)


        obs = OrderedDict([
            ('husky_1', np.concatenate([husky1_pos[2:3], husky1_quat, husky1_vel, husky1_forward_vec, [husky1_move_coeff]])), 
            ('husky_2', np.concatenate([husky2_pos[2:3], husky2_quat, husky2_vel, husky2_forward_vec, [husky2_move_coeff]])),
            ('box_1', np.concatenate([box1_pos - husky1_pos, box1_forward, box_vel])),
            ('box_2', np.concatenate([box2_pos - husky2_pos, box2_forward, box_vel])),
            ('relative_info', np.concatenate([goal1_pos - box1_pos, goal2_pos - box2_pos, [huskys_forward_align_coeff, huskys_right_align_coeff, huskys_dist, goal_box_cos_dist_coeff]]))
        ])


        def ravel(x):
            obs[x] = obs[x].ravel()
        map(ravel, obs.keys())

        return obs

    @property
    def _init_qpos(self):
        # 3 for (x, y, z), 4 for (x, y, z, w), and 2 for each leg
        # If I am correct, one line is for ant1, one line for ant2, the last line is for the box
        '''
        return np.array([0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0, 0., 0.58, 1., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1.,
                         0., 0., 0.8, 1., 0., 0., 0.])
        '''
    
        '''
            The structure of qpose of a single Husky and Box can be found in husky_forward.py _rest()
            
            The first and second 11 elements are for Huskys
            
            The last 7 are for box 
        '''
        return np.array([-2., 1, 0.58, 0., 0., 0., 0., 0., 0, 0., 0.,
                         -2., -1, 0.58, 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0.58, 0., 0., 0., 0.])

    @property
    def _init_qvel(self):
        return np.zeros(self.model.nv)

    def _reset(self):
        '''
        qpos = self._init_qpos + self._init_random(self.model.nq)
        qvel = self._init_qvel + self._init_random(self.model.nv)


        # randomized robots initial positions
        qpos[0:2] = [-2, 1] + np.random.uniform(-1, 1, size=(2,)) * 0.1
        qpos[11:13] = [-2, -1] + np.random.uniform(-1, 1, size=(2,)) * 0.1

        # robot height fixed
        qpos[2] = 0.2
        qpos[13] = 0.2

        init_quat1 = sample_quat(low=-np.pi/10, high=np.pi/10)
        init_quat2 = sample_quat(low=-np.pi/10, high=np.pi/10)

        # randomized robot initial quaternions 
        #qpos[3:7] = [1, 0, 0, 0]
        #qpos[14:18] = [1, 0, 0, 0]
        qpos[3:7] = init_quat1
        qpos[14:18] =init_quat2

        self.set_state(qpos, qvel)

        self._reset_husky_box()

        self._husky1_push = False
        self._husky2_push = False
        self._success_count = 0
        '''
        self._reset_husky_box()

        self._success_count = 0

        return self._get_obs()

    def _reset_husky_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # Initialize box
        '''
        init_box_pos = np.asarray([0, 0, 0.3])
        init_box_quat = sample_quat(low=-np.pi/9, high=np.pi/9)
        qpos[22:25] = init_box_pos
        qpos[25:29] = init_box_quat
        '''
        init_box_pos = np.asarray([-18.5, 0, 0.3])
        init_box_quat = sample_quat(low=0, high=0)
        self._set_pos('box', init_box_pos)
        self._set_quat('box', init_box_quat)

        # Initialize waypoint
        self._set_pos('cpt_1', init_box_pos) 
        self._set_quat('cpt_1', init_box_quat)

        # Initialize robot positions
        husky1_pos = [-20, 1, 0.217]
        husky2_pos = [-20, 1, 0.217] 
        self._set_pos('husky_robot_1', husky1_pos) 
        self._set_pos('husky_robot_2', husky2_pos) 

        # Initialize goal
        #goal_pos = np.asarray([np.random.uniform(low=1.2, high=3), np.random.uniform(low=-1, high=1), 0.3])
        goal_pos = np.asarray([np.random.uniform(low=1, high=-1), np.random.uniform(low=-1, high=1), 0.3])
        goal_quat = sample_quat(low=0, high=0)
        self._set_pos('goal', goal_pos)
        self._set_quat('goal', goal_quat)


    def _render_callback(self):
        lookat = self._get_pos('box')
        cam_pos = lookat + np.array([-4.5, -12, 15])

        cam_id = self._camera_id
        self._set_camera_position(cam_id, cam_pos)
        self._set_camera_rotation(cam_id, lookat)

        self.sim.forward()

