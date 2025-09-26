from __future__ import print_function

import copy
import os

import rospy as ros
import sys
import time
import threading

import numpy as np
import pinocchio as pin
# utility functions
###W
from base_controllers.utils.pidManager import PidManager
from base_controllers.base_controller import BaseController
from base_controllers.utils.math_tools import *

from utils.pid_tuner import PIDTuningGui
from base_controllers.components.whole_body_controller import WholeBodyController
from base_controllers.utils.common_functions import *
from base_controllers.components.inverse_kinematics.inv_kinematics_quadruped import InverseKinematics as AnalyticInverseKinematics
from base_controllers.components.inverse_kinematics.inv_kinematics_pinocchio import robotKinematics as PinocchioInverseKinematics
from base_controllers.components.leg_odometry.leg_odometry import LegOdometry
from termcolor import colored
from std_msgs.msg import Float64MultiArray
import base_controllers.params as conf

from scipy.io import savemat

#gazebo messages
from gazebo_ros import gazebo_interface

from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, WrenchStamped
from sensor_msgs.msg import Imu
from ros_impedance_controller.msg import EffortPid, ExtendedJointState

from base_controllers.components.imu_utils import IMU_utils

import datetime

class QuadrupedController(BaseController):
    def __init__(self, robot_name="hyq", launch_file=None, debug_gui=False):
        super(QuadrupedController, self).__init__(robot_name, launch_file)
        self.qj_0 = conf.robot_params[self.robot_name]['q_0']
        self.dt = conf.robot_params[self.robot_name]['dt']

        self.ee_frames = conf.robot_params[self.robot_name]['ee_frames']
        self.leg_names = [foot[:2] for foot in self.ee_frames]

        self.use_ground_truth_pose = True
        self.debug_gui = debug_gui
        if not self.real_robot:
            self.gravity_comp_duration = 0.5 #1.5
            self.standup_period = 1. #3
        else:
            self.gravity_comp_duration = 1.5
            self.standup_period = 3.


    #####################
    # OVERRIDEN METHODS #
    #####################
    # initVars
    # logData
    # startupProcedure

    def initSubscribers(self):
        # TODO: change into Extendend joint state from anymal
        self.sub_jstate = ros.Subscriber("/legged_odometry/joint_states_forloco", JointState,
                                         callback=self._receive_jstate, queue_size=1, tcp_nodelay=True)
        self.sub_pid_effort = ros.Subscriber("/" + self.robot_name + "/effort_pid", EffortPid,
                                             callback=self._receive_pid_effort, queue_size=1, tcp_nodelay=True)
        self.sub_pose = ros.Subscriber("/legged_odometry/base_state_forloco", Odometry, callback=self._receive_pose,
                                       queue_size=1, tcp_nodelay=True)


        self.sub_contact_lf = ros.Subscriber("/legged_odometry/contact_force_lf_foot", WrenchStamped,
                                                callback=self._receive_contact_lf, queue_size=1, buff_size=2 ** 24,
                                                tcp_nodelay=True)
        self.sub_contact_rf = ros.Subscriber("/legged_odometry/contact_force_rf_foot", WrenchStamped,
                                                callback=self._receive_contact_rf, queue_size=1, buff_size=2 ** 24,
                                                tcp_nodelay=True)
        self.sub_contact_lh = ros.Subscriber("/legged_odometry/contact_force_lh_foot", WrenchStamped,
                                                callback=self._receive_contact_lh, queue_size=1, buff_size=2 ** 24,
                                                tcp_nodelay=True)
        self.sub_contact_rh = ros.Subscriber("/legged_odometry/contact_force_rh_foot", WrenchStamped,
                                                callback=self._receive_contact_rh, queue_size=1, buff_size=2 ** 24,
                                                tcp_nodelay=True)

        # self.sub_imu_lin_acc = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Vector3,
        #                                         callback=self._receive_imu_acc_real, queue_size=1, tcp_nodelay=True)
        # self.sub_imu_euler = ros.Subscriber("/" + self.robot_name + "/euler_imu", Vector3,
        #                                     callback=self._receive_euler, queue_size=1, tcp_nodelay=True)
        # self.sub_pose = ros.Subscriber("/" + self.robot_name + "/ground_truth", Odometry,
        #                                 callback=self._receive_pose_real,
        #                                 queue_size=1, tcp_nodelay=True)
        
    def change_lh_with_rf(self, qj):
        # Change LH and RF for having Pinocchio campatibility
        qj_real = np.copy(qj)
        qj_real[3:6] = qj[6:9]
        qj_real[6:9] = qj[3:6]
        return qj_real

    def _receive_jstate(self, msg):
        for msg_idx in range(len(msg.name)):
            for joint_idx in range(len(self.joint_names)):
                if self.joint_names[joint_idx] == msg.name[msg_idx]:
                    self.q[joint_idx] = msg.position[msg_idx]
                    self.qd[joint_idx] = msg.velocity[msg_idx]
                    self.tau[joint_idx] = msg.effort[msg_idx] 
        
    def _receive_contact_lf(self, msg):
        grf = np.zeros(3)
        grf[0] = msg.wrench.force.x
        grf[1] = msg.wrench.force.y
        grf[2] = msg.wrench.force.z
        self.u.setLegJointState(self.u.leg_map["LF"], grf, self.grForcesLocal_gt)

    def _receive_contact_rf(self, msg):
        grf = np.zeros(3)
        grf[0] = msg.wrench.force.x
        grf[1] = msg.wrench.force.y
        grf[2] = msg.wrench.force.z
        self.u.setLegJointState(self.u.leg_map["RF"], grf, self.grForcesLocal_gt)

    def _receive_contact_lh(self, msg):
        grf = np.zeros(3)
        grf[0] = msg.wrench.force.x
        grf[1] = msg.wrench.force.y
        grf[2] = msg.wrench.force.z
        self.u.setLegJointState(self.u.leg_map["LH"], grf, self.grForcesLocal_gt)

    def _receive_contact_rh(self, msg):
        grf = np.zeros(3)
        grf[0] = msg.wrench.force.x
        grf[1] = msg.wrench.force.y
        grf[2] = msg.wrench.force.z
        self.u.setLegJointState(self.u.leg_map["RH"], grf, self.grForcesLocal_gt)
        
           
    def _receive_imu_acc_real(self, msg):
        # baseLinAccB is with gravity
        self.baseLinAccB[0] = msg.x
        self.baseLinAccB[1] = msg.y
        self.baseLinAccB[2] = msg.z
        # baseLinAccW is without gravity
        self.baseLinAccW = self.b_R_w.T @ (self.baseLinAccB - self.imu_utils.IMU_accelerometer_bias) - self.imu_utils.g0

    def _receive_imu_acc(self, msg):
        self.baseLinAccB[0] = msg.linear_acceleration.x
        self.baseLinAccB[1] = msg.linear_acceleration.y
        self.baseLinAccB[2] = msg.linear_acceleration.z

        self.baseLinAccW = self.b_R_w.T @ (self.baseLinAccB - self.imu_utils.IMU_accelerometer_bias) - self.imu_utils.g0

    def _receive_euler(self, msg):
        self.euler[0] = msg.x
        self.euler[1] = msg.y
        self.euler[2] = msg.z

    def _receive_pose(self, msg):
        self.quaternion[0] = msg.pose.pose.orientation.x
        self.quaternion[1] = msg.pose.pose.orientation.y
        self.quaternion[2] = msg.pose.pose.orientation.z
        self.quaternion[3] = msg.pose.pose.orientation.w

        self.basePoseW[self.u.sp_crd["LX"]] = msg.pose.pose.position.x
        self.basePoseW[self.u.sp_crd["LY"]] = msg.pose.pose.position.y
        self.basePoseW[self.u.sp_crd["LZ"]] = msg.pose.pose.position.z

        self.euler = np.array(euler_from_quaternion(self.quaternion))

        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]

        self.baseTwistW[self.u.sp_crd["LX"]] = msg.twist.twist.linear.x
        self.baseTwistW[self.u.sp_crd["LY"]] = msg.twist.twist.linear.y
        self.baseTwistW[self.u.sp_crd["LZ"]] = msg.twist.twist.linear.z
        self.baseTwistW[self.u.sp_crd["AX"]] = msg.twist.twist.angular.x
        self.baseTwistW[self.u.sp_crd["AY"]] = msg.twist.twist.angular.y
        self.baseTwistW[self.u.sp_crd["AZ"]] = msg.twist.twist.angular.z

        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)




    # def _receive_contact_force_real(self, msg):
    #     #53.0, 82.0, 87.0, 78.0
    #     self.contact_state[0] = msg.data[0] > 73
    #     self.contact_state[1] = msg.data[1] > 102
    #     self.contact_state[2] = msg.data[2] > 107
    #     self.contact_state[3] = msg.data[3] > 98
    #     #print(self.contact_state)


    def _receive_pose_real(self, msg):
        self.quaternion[0] = msg.pose.pose.orientation.x
        self.quaternion[1] = msg.pose.pose.orientation.y
        self.quaternion[2] = msg.pose.pose.orientation.z
        self.quaternion[3] = msg.pose.pose.orientation.w

        self.basePoseW[self.u.sp_crd["LX"]] = self.basePoseW_legOdom[0]
        self.basePoseW[self.u.sp_crd["LY"]] = self.basePoseW_legOdom[1]
        self.basePoseW[self.u.sp_crd["LZ"]] = self.basePoseW_legOdom[2]

        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]

        if False:#any(self.contact_state):
            self.baseTwistW[self.u.sp_crd["LX"]] = self.baseTwistW_legOdom[0]
            self.baseTwistW[self.u.sp_crd["LY"]] = self.baseTwistW_legOdom[1]
            self.baseTwistW[self.u.sp_crd["LZ"]] = self.baseTwistW_legOdom[2]
        else:
            self.baseTwistW[self.u.sp_crd["LX"]] = self.imu_utils.baseLinTwistImuW[0]
            self.baseTwistW[self.u.sp_crd["LY"]] = self.imu_utils.baseLinTwistImuW[1]
            self.baseTwistW[self.u.sp_crd["LZ"]] = self.imu_utils.baseLinTwistImuW[2]

        self.baseTwistW[self.u.sp_crd["AX"]] = msg.twist.twist.angular.x
        self.baseTwistW[self.u.sp_crd["AY"]] = msg.twist.twist.angular.y
        self.baseTwistW[self.u.sp_crd["AZ"]] = msg.twist.twist.angular.z

        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)

    def initVars(self):
        super().initVars()
        self.q_des = np.zeros_like(self.q)

        self.imu_utils = IMU_utils(dt=conf.robot_params[self.robot_name]['dt'])
        #pinocchio based
        self.ikin = PinocchioInverseKinematics(self.robot, conf.robot_params[self.robot_name]['ee_frames'])
        self.IK = AnalyticInverseKinematics(self.robot)
        self.leg_odom = LegOdometry(self.robot, self.real_robot)
        self.legConfig = {}
        if 'solo' in self.robot_name or  self.robot_name == 'hyq' or self.robot_name == 'anymal_d':  # either solo or solo_fw
            self.legConfig['lf'] = ['HipDown', 'KneeInward']
            self.legConfig['lh'] = ['HipDown', 'KneeInward']
            self.legConfig['rf'] = ['HipDown', 'KneeInward']
            self.legConfig['rh'] = ['HipDown', 'KneeInward']

        elif self.robot_name == 'aliengo' or self.robot_name == 'go1' or self.robot_name == 'go2':
            self.legConfig['lf'] = ['HipDown', 'KneeInward']
            self.legConfig['lh'] = ['HipDown', 'KneeOutward']
            self.legConfig['rf'] = ['HipDown', 'KneeInward']
            self.legConfig['rh'] = ['HipDown', 'KneeOutward']

        else:
            assert False, 'leg configuration is not defined for ' + self.robot_name

        self.euler = np.zeros(3)
        # some extra variables

        self.tau_fb = np.zeros(self.robot.na)
        self.tau_ffwd = np.zeros(self.robot.na)
        self.tau_des = np.zeros(self.robot.na)

        self.basePoseW_des = np.zeros(6) * np.nan
        self.baseTwistW_des = np.zeros(6) * np.nan


        self.comPoseW_des = np.zeros(6) * np.nan
        self.comTwistW_des = np.zeros(6) * np.nan

        self.comPosB = np.zeros(3) * np.nan
        self.comVelB = np.zeros(3) * np.nan

        self.basePoseW_legOdom = np.zeros(3) #* np.nan
        self.baseTwistW_legOdom = np.zeros(3) #* np.nan
        ###W
        self.g_mag = np.linalg.norm(self.robot.model.gravity.vector)

        self.grForcesW_des = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesW_wbc = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesB = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesB_ffwd = np.empty(3 * self.robot.nee) * np.nan

        # load gains
        if self.real_robot:
            real_str = '_real'
        else:
            real_str = ''

        # stand alone joint pid
        self.kp_j = conf.robot_params[self.robot_name].get('kp'+real_str, np.zeros(self.robot.na))
        self.kd_j = conf.robot_params[self.robot_name].get('kd'+real_str, np.zeros(self.robot.na))
        self.ki_j = conf.robot_params[self.robot_name].get('ki'+real_str, np.zeros(self.robot.na))

        ###W
        # virtual impedance wrench control
        self.kp_lin = np.diag(conf.robot_params[self.robot_name].get('kp_lin'+real_str, np.zeros(3)))
        self.kd_lin = np.diag(conf.robot_params[self.robot_name].get('kd_lin'+real_str, np.zeros(3)))

        self.kp_ang = np.diag(conf.robot_params[self.robot_name].get('kp_ang'+real_str, np.zeros(3)))
        self.kd_ang = np.diag(conf.robot_params[self.robot_name].get('kd_ang'+real_str, np.zeros(3)))

        # updated in WBC
        self.kp_linW = np.zeros_like(self.kp_lin)
        self.kd_linW = np.zeros_like(self.kd_lin)

        self.kp_angW = np.zeros_like(self.kp_ang)
        self.kd_angW = np.zeros_like(self.kd_ang)

        # joint pid with wbc
        self.kp_wbc_j = conf.robot_params[self.robot_name].get('kp_wbc'+real_str, np.zeros(self.robot.na))
        self.kd_wbc_j = conf.robot_params[self.robot_name].get('kd_wbc'+real_str, np.zeros(self.robot.na))
        self.ki_wbc_j = conf.robot_params[self.robot_name].get('ki_wbc'+real_str, np.zeros(self.robot.na))


        self.wrench_fbW  = np.zeros(6)
        self.wrench_ffW  = np.zeros(6)
        self.wrench_gW   = np.zeros(6)
        self.wrench_gW[self.u.sp_crd["LZ"]] = self.robot.robotMass * self.g_mag
        self.wrench_desW = np.zeros(6)

        self.wrench_fbW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_ffW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_gW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_desW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)

        self.NEMatrix = np.zeros([6, 3*self.robot.nee]) # Newton-Euler matrix

        ###W
        self.wbc = WholeBodyController(conf.robot_params[self.robot_name], self.real_robot, self.robot)

        self.force_th = conf.robot_params[self.robot_name].get('force_th', 0.)
        self.contact_th = conf.robot_params[self.robot_name].get('contact_th', 0.)

        self.W_vel_contacts_des = self.u.full_listOfArrays(4, 3)
        self.B_vel_contacts_des = self.u.full_listOfArrays(4, 3)

        # imu
        self.baseLinAccB = np.full(3, np.nan)
        self.baseLinAccW = np.full(3, np.nan)

        self.solve_time = np.full((1, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comPosB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comVelB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comPoseW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comTwistW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comPoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comTwistW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comVelW_leg_odom = np.full((3), np.nan)
        self.comVelW_leg_odom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)


        self.basePoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseTwistW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.basePoseW_legOdom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseTwistW_legOdom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.tau_fb_log = np.full((self.robot.na, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.tau_des_log = np.full((self.robot.na, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)


        self.grForcesB_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.grForcesW_gt_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.grForcesW_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.grForcesW_wbc_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.W_contacts_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.W_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.B_contacts_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.B_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.B_vel_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.W_vel_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.contact_state_log = np.full((self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.baseLinAccW_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseLinAccB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.baseLinTwistImuW_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        # robot height is the height of the robot base frame in home configuration
        self.robot_height = 0.

        neutral_fb_jointstate = np.hstack([pin.neutral(self.robot.model)[0:7], self.u.mapToRos(conf.robot_params[self.robot_name]['q_0']) ])

        self.robot.forwardKinematics(neutral_fb_jointstate)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        for id in self.robot.getEndEffectorsFrameId:
            self.robot_height += self.robot.data.oMf[id].translation[2]
        self.robot_height /= -4.
        print(colored(f"Robot height correspontent to q0 configuration is {self.robot_height+0.02}","red")) #Includes the foot radius
        self.loop_time_log = np.full((conf.robot_params[self.robot_name]['buffer_size']), np.nan)

        half_lenght = self.robot.collision_model.geometryObjects[0].geometry.halfSide[0]
        half_width = self.robot.collision_model.geometryObjects[0].geometry.halfSide[1]
        half_height = self.robot.collision_model.geometryObjects[0].geometry.halfSide[2]
        self.bControlPoints = [ np.array([half_lenght, half_width, -half_height]),# lf_bottom
                                np.array([-half_lenght, half_width, -half_height])  ,# lh_bottom
                                np.array([half_lenght, -half_width, -half_height])  ,# rf_bottom
                                np.array([-half_lenght, -half_width, -half_height]) , # rf_bottom
                                np.array([half_lenght, half_width, half_height])  ,# lf_top
                                np.array([-half_lenght, half_width, half_height]) , # lh_top
                                np.array([half_lenght, -half_width, half_height]) , # rf_top
                                np.array([-half_lenght, -half_width, half_height])]  # rf_top

        self.kfe_idx = [self.robot.model.getFrameId(leg + '_kfe_joint') for leg in ['lf', 'lh', 'rf', 'rh']]

        # self.pid_tuning_gui = PIDTuningGui(self, init_freq=0.5, real_robot_=self.real_robot)

        if self.debug_gui:
            self.stop_thread = False
            self.thread_pid = threading.Thread(target=self.pid_tuning_gui.init_pid_tuning_ui)
            self.thread_pid.daemon = True
            self.thread_pid.start()

    def logData(self):
        # full with new values
        self.comPosB_log[:, self.log_counter] = self.comPosB
        self.comVelB_log[:, self.log_counter] = self.comVelB
        self.comPoseW_log[:, self.log_counter] = self.comPoseW
        self.comTwistW_log[:, self.log_counter] = self.comTwistW
        self.comPoseW_des_log[:, self.log_counter] = self.comPoseW_des
        self.comTwistW_des_log[:, self.log_counter] = self.comTwistW_des
        self.basePoseW_log[:, self.log_counter] = self.basePoseW
        self.baseTwistW_log[:, self.log_counter] = self.baseTwistW
        self.basePoseW_des_log[:, self.log_counter] = self.basePoseW_des
        self.baseTwistW_des_log[:, self.log_counter] = self.baseTwistW_des
        self.basePoseW_legOdom_log[:, self.log_counter] = self.basePoseW_legOdom
        self.baseTwistW_legOdom_log[:, self.log_counter] = self.baseTwistW_legOdom
        self.q_des_log[:, self.log_counter] = self.q_des
        self.q_log[:, self.log_counter] = self.q
        self.qd_des_log[:, self.log_counter] = self.qd_des
        self.qd_log[:, self.log_counter] = self.qd
        self.tau_fb_log[:, self.log_counter] = self.tau_fb
        self.tau_ffwd_log[:, self.log_counter] = self.tau_ffwd

        self.tau_des = self.tau_ffwd + self.tau_fb
        self.tau_des_log[:, self.log_counter] = self.tau_des
        self.tau_log[:, self.log_counter] = self.tau
        self.grForcesW_log[:, self.log_counter] = self.grForcesW
        self.grForcesW_des_log[:, self.log_counter] = self.grForcesW_des
        self.grForcesW_wbc_log[:, self.log_counter] = self.grForcesW_wbc
        self.grForcesW_gt_log[:, self.log_counter] = self.grForcesW_gt
        self.grForcesB_log[:, self.log_counter] = self.grForcesB
        self.contact_state_log[:, self.log_counter] = self.contact_state

        self.baseLinAccW_log[:, self.log_counter] = self.baseLinAccW
        self.baseLinAccB_log[:, self.log_counter] = self.baseLinAccB

        self.comVelW_leg_odom_log[:, self.log_counter] = self.comVelW_leg_odom

        for leg in range(4):
            start = 3 * leg
            end = 3 * (leg+1)
            self.B_contacts_log[start:end, self.log_counter] = self.B_contacts[leg]
            self.B_contacts_des_log[start:end, self.log_counter] = self.B_contacts_des[leg]

            self.W_contacts_log[start:end, self.log_counter] = self.W_contacts[leg]
            self.W_contacts_des_log[start:end, self.log_counter] = self.W_contacts_des[leg]

            self.B_vel_contacts_des_log[start:end, self.log_counter] = self.B_vel_contacts_des[leg]
            self.W_vel_contacts_des_log[start:end, self.log_counter] = self.W_vel_contacts_des[leg]

        self.baseLinTwistImuW_log[:, self.log_counter] = self.imu_utils.baseLinTwistImuW

        ###W
        self.wrench_fbW_log[:, self.log_counter] = self.wrench_fbW
        self.wrench_ffW_log[:, self.log_counter] = self.wrench_ffW
        self.wrench_gW_log[:, self.log_counter] = self.wrench_gW
        self.wrench_desW_log[:, self.log_counter] = self.wrench_desW
###W

        self.time_log[self.log_counter] = self.time
        self.loop_time_log[self.log_counter] = self.loop_time

        self.log_counter += 1
        self.log_counter %= conf.robot_params[self.robot_name]['buffer_size']
    
    def check_faulty_ping(self, ip="192.168.1.1"):
        response = os.system("ping -c 1 " + ip)
        # and then check the response...
        if response == 0:
            pingstatus = f"Robot Network Active: pinging {ip} successful"
        else:
            pingstatus = f"Robot Network not Active, cannot ping {ip}, create a local network with gateway {ip}"
        print(colored(pingstatus, "red"))
        return response

    def startController(self, world_name=None, xacro_path=None, use_ground_truth_pose=True, use_ground_truth_contacts=True, additional_args=[]):

        if self.real_robot == False:
            self.use_ground_truth_pose = use_ground_truth_pose
            self.use_ground_truth_contacts = use_ground_truth_contacts
        else:
            self.use_ground_truth_pose = False
            self.use_ground_truth_contacts = False
            if self.check_faulty_ping(conf.robot_params[self.robot_name]['ip']):
                sys.exit()

        self.start()                               # as a thread

        self.go0_conf = 'home'
        if additional_args is not None:
            for arg in additional_args:
                if 'go0_conf:=' in arg:
                    self.go0_conf = arg.replace('go0_conf:=', '')


        additional_args.append("load_force_sensors:="+str(not self.use_ground_truth_contacts).lower())
        if self.use_ground_truth_contacts and (world_name is None or not 'slow' in world_name):
            print('Cannot use ground truth contact with not slow world file')
            print('Set world file to slow.world')
            world_name = 'slow.world'

        # self.startSimulator(world_name=world_name, additional_args=additional_args)            # run gazebo
        if world_name is None:
            self.world_name_str = ''
        else:
            self.world_name_str = world_name
        if 'camera' in self.world_name_str:
            # check if some old jpg are still in /tmp
            # this command prevent for Argument list too long in bash http://mywiki.wooledge.org/BashFAQ/095
            print(colored('Removing jpg files', 'blue'), flush=True)
            remove_jpg_cmd = 'for f in /tmp/camera_save/*; do rm "$f"; done'
            os.system(remove_jpg_cmd)
            print(colored('Jpg files removed', 'blue'), flush=True)

        self.loadModelAndPublishers(xacro_path)    # load robot and all the publishers
        #self.resetGravity(True)
        self.initVars()                            # overloaded method
        self.initSubscribers()
        self.rate = ros.Rate(1 / self.dt)
        print(colored("Started QuadrupedController", "blue"))

    
    def loadModelAndPublishers(self, xacro_path=None):

        # Loading a robot model of robot (Pinocchio)
        if xacro_path is None:
            xacro_path = rospkg.RosPack().get_path(
                self.robot_name + '_description') + '/robots/' + self.robot_name + '.urdf.xacro'
        else:
            print("loading custom xacro path: ", xacro_path)
        self.robot = self.getRobotModelFloating(self.robot_name)
        self.pub_des_jstate = ros.Publisher("/locosim/des_joint_state", JointState, queue_size=10)

        self.broadcaster = SafeTFBroadcaster()

    def getRobotModelFloating(self, robot_name="hyq"):
        ERROR_MSG = 'You should set the environment variable LOCOSIM_DIR"\n'
        path = os.environ.get('LOCOSIM_DIR', ERROR_MSG)
        urdf_location = path + "/robot_urdf/generated_urdf/" + robot_name + ".urdf"
        print(path)
        print(urdf_location)
        robot = RobotWrapper.BuildFromURDF(urdf_location, root_joint=pinocchio.JointModelFreeFlyer())
    
        return robot

    def self_weightCompensation(self):
        # require the call to updateKinematics
        gravity_torques = np.zeros(12)#self.g_joints
        return gravity_torques

    def gravityCompensation(self):
        # require the call to updateKinematics
        return self.WBC(des_pose = None, des_twist = None, des_acc = None, comControlled = True, type = 'projection')
  
    def support_poly(self, contacts):
        # Wcontacts: lf, rf, lh, rh
        sp = {}
        # CCW order
        sides_order = {'F': [1, 0], 'L': [0, 2], 'H': [2, 3], 'R': [3, 1]}
        for side in sides_order: # side is the key of sides_order
            p0 = contacts[sides_order[side][0]][0:2]
            p1 = contacts[sides_order[side][1]][0:2]
            m, q = self.line2points2D(p0, p1)
            sp['line'+side] = {'m': m, 'q': q, 'p0':p0, 'p1':p1}
        return sp

    @staticmethod
    def line2points2D(p0, p1):
        # line defined as y= mx+q
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        q = p0[1] - m * p0[0]
        return m, q

    def send_command(self, q_des=None, qd_des=None, tau_ffwd=None, log_data_in_send_command = False):
        # q_des, qd_des, and tau_ffwd have dimension 12
        # and are ordered as on the robot

        if q_des is not None:
            self.q_des = q_des

        if qd_des is not None:
            self.qd_des = qd_des

        if tau_ffwd is not None:
            self.tau_ffwd = tau_ffwd

        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        # log variables
        if log_data_in_send_command:
            self.logData()
        self.rate.sleep()
        self.sync_check()
        self.time = np.round(self.time + self.dt, 4)#np.array([self.loop_time]), 3)


    def visualizeContacts(self, delete_markers=False):
        for legid in self.u.leg_map.keys():

            leg = self.u.leg_map[legid]
            if self.contact_state[leg]:
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW/ (6*self.robot.robotMass)),
                                       "green")
                
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW_des/ (6*self.robot.robotMass)),
                                       "blue")
                
                #self.ros_pub.add_marker(self.W_contacts[leg], radius=0.1)
            else:
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       np.zeros(3),
                                       "green", scale=0.0001)
                #self.ros_pub.add_marker(self.W_contacts[leg], radius=0.001)

            if (self.use_ground_truth_contacts):
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW_gt / (6 * self.robot.robotMass)),
                                       "red")


        # self.ros_pub.add_polygon([self.B_contacts[0],
        #                           self.B_contacts[1],
        #                           self.B_contacts[3],
        #                           self.B_contacts[2],
        #                           self.B_contacts[0] ], "red", visual_frame="base_link")
        #
        # self.ros_pub.add_polygon([self.B_contacts_des[0],
        #                           self.B_contacts_des[1],
        #                           self.B_contacts_des[3],
        #                           self.B_contacts_des[2],
        #                           self.B_contacts_des[0]], "green", visual_frame="base_link")
        #

        self.ros_pub.publishVisual(delete_markers=delete_markers)

    def updateKinematics(self, update_legOdom=True, noise=None):
        if noise is not None:
            if 'qd' in noise:
                self.qd += noise['qd'].draw()
            if 'tau' in noise:
                self.tau += noise['tau'].draw()
        self.basePoseW_legOdom, self.baseTwistW_legOdom = self.leg_odom.base_in_world(contact_state=self.contact_state,
                                                                                      B_contacts=self.B_contacts,
                                                                                      b_R_w=self.b_R_w,
                                                                                      wJ=self.wJ,
                                                                                      ang_vel=self.u.angPart(self.baseTwistW),
                                                                                      qd=self.qd,
                                                                                      update_legOdom=update_legOdom)
        self.imu_utils.compute_lin_vel(self.baseLinAccW, self.loop_time)
        super(QuadrupedController, self).updateKinematics()

    def checkBaseCollisions(self):
        # base control points
        for i, pt in enumerate(self.bControlPoints):
            wControlPoint = self.mapBaseToWorld(pt)
            if wControlPoint[2] < self.contact_th:
                return True
        return False

    def checkKFECollisions(self):
        # kfe collision
        for i, id in enumerate(self.kfe_idx):
            wKfe_pos = self.mapBaseToWorld(self.robot.data.oMf[id].translation)  # update kin computes quantity in base frame
            if wKfe_pos[2] < self.contact_th:
                return True
        return False

    def checkGroundCollisions(self):
        # retrun codes
        # -1 no collisions
        # base collisions
        # 0 lf_bottom
        # 1 lh_bottom
        # 2 rf_bottom
        # 3 rf_bottom
        # 4 lf_top
        # 5 lh_top
        # 6 rf_top
        # 7 rf_top
        # kfe collisions
        # 8 lf_kfe
        # 9 lf_kfe
        # 10 lf_kfe
        # 11 lf_kfe

        # return True/False

        # base control points
        for i, pt in enumerate(self.bControlPoints):
            wControlPoint = self.mapBaseToWorld(pt)
            if wControlPoint[2] < self.contact_th:
                #return i
                return True

        # kfe collision
        for i, id in enumerate(self.kfe_idx):
            wKfe_pos = self.mapBaseToWorld(self.robot.data.oMf[id].translation) # update kin computes quantity in base frame
            if wKfe_pos[2] < self.contact_th:
                #return self.bControlPoints+i
                return True

        # return -1
        return False

    def customStartupProcedure(self):
        print(colored("Custom Startup Procedure", "red"))
        self.q_des = self.qj_0
        # self.pid = PidManager(self.joint_names)
        # # set joint pdi gains
        # self.pid.setPDjoints(conf.robot_params[self.robot_name]['kp'],
        #                     conf.robot_params[self.robot_name]['kd'],
        #                     conf.robot_params[self.robot_name]['ki'])
        # p.resetRobot(basePoseDes=np.array([0, 0, conf.robot_params[self.robot_name]['spawn_z'],  0., 0., 0.]))
        while self.time <= self.startTrust:
            self.updateKinematics()
            # self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensation(self.W_contacts, self.wJ, self.h_joints,
            #                                                         self.basePoseW, self.comPoseW)
            self.send_command(self.q_des, self.qd_des, self.tau_ffwd)    

    def send_des_jstate(self, q_des, qd_des, tau_ffwd, soft_limits = 0.9, clip_commands = False):
         # No need to change the convention because in the HW interface we use our conventtion (see ros_impedance_contoller_xx.yaml)
         msg = JointState()
         msg.header.stamp = ros.Time.now()
         msg.name = self.joint_names
         if clip_commands:
             msg.position = np.clip(q_des,self.robot.model.lowerPositionLimit[-self.robot.na:] * soft_limits ,self.robot.model.upperPositionLimit[-self.robot.na:] * soft_limits)
             msg.velocity = np.clip(qd_des, -self.robot.model.velocityLimit[-self.robot.na:] * soft_limits ,self.robot.model.velocityLimit[-self.robot.na:] * soft_limits)
             msg.effort = np.clip(tau_ffwd, -self.robot.model.effortLimit[-self.robot.na:] * soft_limits ,self.robot.model.effortLimit[-self.robot.na:] * soft_limits)
         else:
             msg.position = self.change_lh_with_rf(q_des)
             msg.velocity = self.change_lh_with_rf(qd_des)
             msg.effort = self.change_lh_with_rf(tau_ffwd)
         self.pub_des_jstate.publish(msg)

         

    def computeJcb(self, feetW, com, stance_legs):
        Jb = np.zeros([3 * self.robot.nee, 6])  # Newton-Euler matrix
        for leg in range(self.robot.nee):
            start_row = 3 * leg
            end_row = 3 * (leg + 1)
            if stance_legs[leg]:
                # ---> linear part
                # identity matrix (I avoid to rewrite zeros)
                Jb[start_row:end_row, :3] = np.identity(3)
                # ---> angular part
                # all in a function
                Jb[start_row:end_row, 3:] = -pin.skew(feetW[leg] - com)
            else:
                Jb[start_row:end_row, 3:] = np.zeros(3)
                Jb[start_row:end_row, :3] = np.zeros(3)
        return Jb

    def evalThrust(self, t_, freq, amp_lin):

        com = np.array([
            self.initial_com[0],
            self.initial_com[1] + amp_lin[1] * np.cos(2 * np.pi * freq * t_),
            self.initial_com[2] + amp_lin[2] * np.sin(2 * np.pi * freq * t_),
        ])

        comd = np.array([
            0.,
            -2 * np.pi * freq * amp_lin[1] * np.sin(2 * np.pi * freq * t_),
            2 * np.pi * freq * amp_lin[2] * np.cos(2 * np.pi * freq * t_),
        ])

        comdd = np.array([
            0.,
            -(2 * np.pi * freq * amp_lin[1])**2 * np.cos(2 * np.pi * freq * t_),
            -(2 * np.pi * freq * amp_lin[2])**2 * np.sin(2 * np.pi * freq * t_),
        ])
        
        eul = np.array([0., 0.0, 0]) 
        euld, euldd = np.copy(eul), np.copy(eul)

        # com = self.initial_com + np.multiply(amp_lin, np.sin(2*np.pi*freq * t_))
        # comd = np.multiply(2*np.pi*freq*amp_lin,  np.cos(2*np.pi*freq * t_))
        # comdd = np.multiply(np.power(2*np.pi*freq*amp_lin, 2), -np.sin(2*np.pi*freq * t_))

        # eul = np.array([0., 0.0, 0]) + np.multiply(amp_ang, np.sin(2 * np.pi * freq * t_))
        # euld = np.multiply(2 * np.pi * freq * amp_ang, np.cos(2 * np.pi * freq * t_))
        # euldd = np.multiply(np.power(2 * np.pi * freq * amp_ang, 2), -np.sin(2 * np.pi * freq * t_))

        Jb = p.computeJcb(self.W_contacts_sampled, com, self.stance_legs)

        W_des_basePose = np.empty(6)
        W_des_basePose[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = com
        W_des_basePose[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = eul

        W_des_baseTwist = np.empty(6)
        W_des_baseTwist[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = comd
        Jomega = self.math_utils.Tomega(eul)
        W_des_baseTwist[self.u.sp_crd['AX']:self.u.sp_crd['AX'] +
                        3] = self.math_utils.Tomega(eul).dot(euld)

        W_des_baseAcc = np.empty(6)
        W_des_baseAcc[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = comdd
        # compute w_omega_dot =  Jomega* euler_rates_dot + Jomega_dot*euler_rates (Jomega already computed, see above)
        Jomega_dot = self.math_utils.Tomega_dot(eul, euld)
        W_des_baseAcc[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega @ euldd + Jomega_dot @ euld

        # map base twist into feet relative vel (wrt com/base)
        W_feetRelVelDes = -Jb.dot(W_des_baseTwist)
        w_R_b_des = self.math_utils.eul2Rot(eul)

        grf_ffwd = np.zeros(12)
        tau_ffwd = np.zeros(12)
        qd_des = np.zeros(12)
        q_des = np.zeros(12)
        fbjoints = pin.neutral(self.robot.model)
        w_J = self.u.listOfArrays(4, np.zeros((3, 3)))
        # integrate relative Velocity

        for leg in range(self.robot.nee):
            # with this you do not have proper tracking of com and trunk orientation, I think there is a bug in the ik
            # self.W_feetRelPosDes[leg] += W_feetRelVelDes[3 * leg:3 * (leg+1)]*self.dt
            # this has better tracking
            # should use desired values to generate traj otherwise if it is unstable it detroys the ref signal
            self.W_feetRelPosDes[leg] = self.W_contacts_sampled[leg] - com

            # q_des[3 * leg:3 * (leg+1)], isFeasible = self.IK.ik_leg(w_R_b_des.T.dot(self.W_feetRelPosDes[leg]),
            #                                                         self.leg_names[leg],
            #                                                         self.legConfig[self.leg_names[leg]][0],
            #                                                         self.legConfig[self.leg_names[leg]][1])
            q_des[3 * leg:3 * (leg+1)], _ = self.ikin.footInverseKinematicsFixedBaseLineSearch(
                w_R_b_des.T.dot(self.W_feetRelPosDes[leg]),
                self.ee_frames[leg],
                self.q[3 * leg:3 * (leg+1)]
            )
            # for joint velocity we need to recompute the Jacobian (in the world frame) for the computed joint position q_des
            # you need to fill in also the floating base part
            quat_des = pin.Quaternion(w_R_b_des)
            fbjoints[:3] = com
            fbjoints[3:7] = np.array([quat_des.x, quat_des.y, quat_des.z, quat_des.w])
            fbjoints[7:] = q_des

            pin.forwardKinematics(self.des_robot.model, self.des_robot.data, fbjoints, np.zeros(
                self.des_robot.model.nv),   np.zeros(self.des_robot.model.nv))
            pin.computeJointJacobians(
                self.des_robot.model, self.des_robot.data)
            pin.computeFrameJacobian(self.des_robot.model, self.des_robot.data,
                                    fbjoints, p.des_robot.model.getFrameId(self.ee_frames[leg]))
            w_J[leg] = pin.getFrameJacobian(self.des_robot.model, self.des_robot.data,
                                            p.des_robot.model.getFrameId(
                                                self.ee_frames[leg]),
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, 6 + leg * 3:6 + leg * 3 + 3]
            # compute joint variables
            qd_des[3 * leg:3 * (leg+1)] = np.linalg.pinv(w_J[leg]).dot(W_feetRelVelDes[3 * leg:3 * (leg+1)])
            
            tau_ffwd, self.grForcesW_des = self.wbc.computeWBC(self.W_contacts, self.wJ, self.h_joints,  self.basePoseW, self.comPoseW, self.baseTwistW, self.comTwistW,
                                                        W_des_basePose, W_des_baseTwist, W_des_baseAcc, self.centroidalInertiaB,
                                                        comControlled=False, type='projection', stance_legs=self.stance_legs)
            
        return q_des, qd_des, tau_ffwd, W_des_basePose, W_des_baseTwist


if __name__ == '__main__':
    p = QuadrupedController('anymal_d', debug_gui=False)
    world_name = 'fast.world'
    use_gui = False
    
    try:
        #p.startController(world_name='slow.world')
        ros.init_node("loco_node")
        p.startController(world_name=world_name,
                          use_ground_truth_pose=True,
                          use_ground_truth_contacts=False,
                          additional_args=['gui:='+str(use_gui),
                                           'go0_conf:=standDown'])

        p.des_robot = RobotWrapper.BuildFromURDF(
            os.environ.get('LOCOSIM_DIR') + "/robot_urdf/generated_urdf/" + p.robot_name + ".urdf",
            root_joint=pinocchio.JointModelFreeFlyer())

        p.startTrust = 3.
        p.startMotion = 6. # it starts after 3 second of the prev command
        p.customStartupProcedure()

        # initial pose (we consider com the base origin)
        com_0 = p.basePoseW[:3].copy()
        eul_0 = p.basePoseW[3:].copy()

        p.initial_com = np.copy(com_0)
        print(f"Initial Com Position is {p.initial_com}")
        print(f"Initial Joint Position is {p.q}")
        print(f"Initial Joint torques {p.tau_ffwd}")
        print(colored(f"IMPORTANT: you cannot control both pitch and Z and expect 0 error on comX, only on base, because it is an impossible task!", "red"))
        p.T_th = np.inf
        p.T_th_total = np.inf

        # p.computeHeuristicSolutionBezierLinear(com_0, p.com_lo_b, p.comd_lo_b, p.T_th_b)
        # # we have the explosive part only for the lineat part
        # p.computeHeuristicSolutionBezierAngular(eul_0, p.eul_lo, p.euld_lo, p.T_th_total)

        # # this is for visualization
        # if not p.DEBUG:
        #     p.computeTrajectoryBezier(p.T_th_b, p.com_lo_e)
        p.stance_legs = [True, True, True, True]

        # reset integration of feet
        p.W_feetRelPosDes = np.copy(p.W_contacts - com_0)
        p.W_contacts_sampled = np.copy(p.W_contacts)

        # if not p.real_robot:
        #     p.setSimSpeed(dt_sim=0.001, max_update_rate=300, iters=1500)

        # Topic that send des pos vel to anymal simulator
        

        
        while not ros.is_shutdown():
            p.updateKinematics()

            if p.time < p.startMotion:
                p.tau_ffwd, p.grForcesW_des = p.wbc.gravityCompensationBase(p.B_contacts,
                                                                                p.wJ,
                                                                                p.h_joints,
                                                                            p.comPoseW)
                p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)

            else:
                t = p.time - p.startMotion

                ellipse_f = 0.8
                ellipse_amp_lin = np.array([0., 0.05, 0.02])

                start = time.time()
                p.q_des, p.qd_des, p.tau_ffwd, p.basePoseW_des, p.baseTwistW_des = p.evalThrust(t.item(), ellipse_f, ellipse_amp_lin)
                p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
                p.solve_time[:, p.log_counter] = time.time() - start
                
            # p.visualizeContacts()
            p.logData()

            p.rate.sleep()
            p.sync_check()

            p.time = np.round(p.time + p.dt, 4)
        

    except (ros.ROSInterruptException, ros.service.ServiceException):
        ros.signal_shutdown("killed")
        os.system(" rosnode kill /"+p.robot_name+"/ros_impedance_controller")    
        os.system(" rosnode kill /gazebo")
        os.system("pkill rosmaster")
        


    if conf.plotting:
        
        # Joint
        plotJoint('position', time_log=p.time_log, q_log=p.q_log, q_des_log=p.q_des_log, sharex=True, sharey=False,
                  start=0, end=-1)
        plotJoint('velocity', time_log=p.time_log, qd_log=p.qd_log, qd_des_log=p.qd_des_log, sharex=True, sharey=False,
                  start=0, end=-1)
        
        # Base
        plotFrame('position', time_log=p.time_log, des_Pose_log=p.basePoseW_des_log, Pose_log=p.basePoseW_log,
                  title='Base', frame='W', sharex=True, sharey=False, start=0, end=-1)
        plotFrame('velocity', time_log=p.time_log, des_Twist_log=p.baseTwistW_des_log, Twist_log=p.baseTwistW_log,
                  title='Base', frame='W', sharex=True, sharey=False, start=0, end=-1)

        time_log = p.solve_time[~np.isnan(p.solve_time)]
        # print(time_log)
        sort_time_log = np.sort(time_log)
        cp_time = np.arange(1, len(sort_time_log) + 1) / len(sort_time_log)
        # print(cp_time)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.step(sort_time_log, cp_time, c='b', where='post')
        plt.axhline(0.99, c='black', ls='--', lw=2)
        plt.axvline(np.quantile(time_log, 0.99), c='purple', ls='--', lw=2)
        plt.xlabel("Computation time (s)")
        plt.ylabel("Cumulative Percentage")
        plt.ylim(0, 1.05)
        plt.xlim(0, 0.01)  
        plt.legend(framealpha=1.0)


        plt.figure()
        plt.plot(p.basePoseW_log[1], p.basePoseW_log[2], c='blue', label='Real')
        plt.plot(p.basePoseW_des_log[1], p.basePoseW_des_log[2], c='red', label='Ref')
        plt.xlabel('y (m)')
        plt.ylabel('z (m)')
        plt.legend(framealpha=1.0)