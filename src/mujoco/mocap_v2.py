#!/usr/bin/env python3
import os
import json
import math
import copy
import numpy as np
from os import getcwd
from pyquaternion import Quaternion
from mujoco.mocap_util import align_position, align_rotation
from mujoco.mocap_util import BODY_JOINTS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_DEFS

from transformations import euler_from_quaternion, quaternion_from_euler

class MocapDM(object):
    def __init__(self, robot):
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim = 4
        self.robot = robot
        # mocap data
        self.dt = None
        self.data_config = None
        self.data_vel = None
        self.data_body_xpos = None
        self.data_geom_xpos = None

    def load_mocap(self, filepath):
        self.read_raw_data(filepath)

    def read_raw_data(self, filepath):
        motions = None
        self.loop = None
        all_states = []

        durations = []

        with open(filepath, 'r') as fin:
            data = json.load(fin)

        motions = np.array(data["Frames"])
        try:
            self.loop = data["Loop"]
        except:
            print("Warning: No loop information found in mocap file")
        self.dt = motions[0][0]

        if "Format" not in data:
            m_shape = np.shape(motions)
            self.data = np.full(m_shape, np.nan)

            total_time = 0.0
            for each_frame in motions:
                duration = each_frame[0]
                each_frame[0] = total_time
                total_time += duration
                durations.append(duration)

                curr_idx = 1
                offset_idx = 8
                state = {}
                state['root_pos'] = align_position(each_frame[curr_idx:curr_idx+3])
                # state['root_pos'][2] += 0.08
                state['root_rot'] = align_rotation(each_frame[curr_idx+3:offset_idx])
                for each_joint in BODY_JOINTS_IN_DP_ORDER:
                    curr_idx = offset_idx
                    dof = DOF_DEF[each_joint]
                    if dof == 1:
                        offset_idx += 1
                        state[each_joint] = each_frame[curr_idx:offset_idx]
                    elif dof == 3:
                        offset_idx += 4
                        state[each_joint] = align_rotation(each_frame[curr_idx:offset_idx])
                all_states.append(state)

            self.all_states = all_states
            self.durations = durations

            # convert
            self.data_vel = []
            self.data_config = []

            for k in range(len(self.all_states)):
                tmp_vel = []
                tmp_angle = []
                state = self.all_states[k]
                if k == 0:
                    dura = self.durations[k]
                else:
                    dura = self.durations[k-1]

                # time duration
                init_idx = 0
                offset_idx = 1
                self.data[k, init_idx:offset_idx] = dura

                # root pos
                init_idx = offset_idx
                offset_idx += 3
                self.data[k, init_idx:offset_idx] = np.array(state['root_pos'])
                if k == 0:
                    tmp_vel += [0.0, 0.0, 0.0]
                else:
                    tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
                tmp_angle += state['root_pos'].tolist()

                # root rot
                init_idx = offset_idx
                offset_idx += 4
                self.data[k, init_idx:offset_idx] = np.array(state['root_rot'])
                if k == 0:
                    tmp_vel += [0.0, 0.0, 0.0]
                else:
                    tmp_vel += self.calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
                tmp_angle += state['root_rot'].tolist()

                for each_joint in BODY_JOINTS:
                    init_idx = offset_idx
                    tmp_val = state[each_joint]
                    if DOF_DEF[each_joint] == 1:
                        assert 1 == len(tmp_val)
                        offset_idx += 1
                        self.data[k, init_idx:offset_idx] = state[each_joint]
                        if k == 0:
                            tmp_vel += [0.0]
                        else:
                            tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
                        tmp_angle += state[each_joint].tolist()
                    elif DOF_DEF[each_joint] == 3:
                        assert 4 == len(tmp_val)
                        offset_idx += 4
                        self.data[k, init_idx:offset_idx] = state[each_joint]
                        if k == 0:
                            tmp_vel += [0.0, 0.0, 0.0]
                        else:
                            tmp_vel += self.calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
                        quat = state[each_joint]
                        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                        euler_tuple = euler_from_quaternion(quat, axes='rxyz')
                        tmp_angle += list(euler_tuple)
                        ## For testing
                        # quat_after = quaternion_from_euler(euler_tuple[0], euler_tuple[1], euler_tuple[2], axes='rxyz')
                        # np.set_printoptions(precision=4, suppress=True)
                        # diff = quat-quat_after
                        # if diff[3] > 0.5:
                        #     import pdb
                        #     pdb.set_trace()
                        #     print(diff)
                self.data_vel.append(np.array(tmp_vel))
                self.data_config.append(np.array(tmp_angle))
        else:
            self.data_config = motions[:, 1:]
            # calculate velocity
            self.data_vel = []
            for k in range(len(self.data_config)):
                kp = k - 1
                if k == 0:
                    kp = 0
                prev_root_xyz = self.data_config[kp][:3]
                prev_root_qxyz = self.data_config[kp][3:7]
                prev_rest = self.data_config[kp][7:]
                next_root_xyz = self.data_config[k][:3]
                next_root_qxyz = self.data_config[k][3:7]
                next_rest = self.data_config[k][7:]
                vel_root_xyz = (next_root_xyz - prev_root_xyz) / self.dt
                vel_root_qxyz = self.calc_rot_vel(prev_root_qxyz, next_root_qxyz, self.dt)
                vel_rest = (next_rest - prev_rest) / self.dt
                self.data_vel.append(np.concatenate([vel_root_xyz, vel_root_qxyz, vel_rest]))

        
        if True:
            self.data_body_xpos = []
            self.data_geom_xpos = []
            # C.o.M:
            # model.body_mass
            # data.body_xpos
            # End effectors
            # data.geom_xpos
            # model.geom_name2id
            from deepmimic_env import DPEnv
            env = DPEnv(load_mocap=False, robot=self.robot)
            env.reset_model()
            for qpos, qvel in zip(self.data_config, self.data_vel):
                env.set_state(qpos, qvel)
                self.data_body_xpos.append(np.array(env.sim.data.body_xpos)*1.)
                self.data_geom_xpos.append(np.array(env.sim.data.geom_xpos)*1.)

        # interpolate frames to match target_dt
        # e.g. if target dt is 0.0166 and current dt is 0.0333, then we need to interpolate x2
        target_dt = 0.01666 # simulator dt
        dt_tolerance = 0.1 # 10% tolerance away from integer ratio
        dt_ratio = self.dt / target_dt
        dt_int_ratio = int(dt_ratio)
        if abs(dt_ratio - dt_int_ratio) > dt_tolerance:
            raise Exception("Invalid dt ratio, cannot interpolate mocap frames: %f"%(dt_ratio))
        if dt_int_ratio > 1:
            add_frames = dt_int_ratio - 1
            new_dt = target_dt
            new_data_config = []
            new_data_vel = []
            new_data_body_xpos = []
            new_data_geom_xpos = []
            for ia, ib in zip(range(len(self.data_config)-1), range(1, len(self.data_config))):
                for k in range(dt_int_ratio):
                    B = k * 1.0 / dt_int_ratio
                    A = 1.0 - B
                    new_data_config.append(A * self.data_config[ia] + B * self.data_config[ib])
                    new_data_vel.append(A * self.data_vel[ia] + B * self.data_vel[ib])
                    new_data_body_xpos.append(A * self.data_body_xpos[ia] + B * self.data_body_xpos[ib])
                    new_data_geom_xpos.append(A * self.data_geom_xpos[ia] + B * self.data_geom_xpos[ib])
            self.dt = new_dt
            self.data_config = new_data_config
            self.data_vel = new_data_vel
            self.data_body_xpos = new_data_body_xpos
            self.data_geom_xpos = new_data_geom_xpos


    def calc_rot_vel(self, seg_0, seg_1, dura):
        q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
        q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

        q_diff =  q_0.conjugate * q_1
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_diff = angle/dura * axis
        diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

        return diff_angular

    def play(self, mocap_filepath):
        from mujoco_py import load_model_from_xml, MjSim, MjViewer

        curr_path = getcwd()
        xmlpath = '/mujoco/humanoid_deepmimic/envs/asset/dp_env_v2.xml'
        with open(curr_path + xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)

        self.read_raw_data(mocap_filepath)

        from time import sleep

        phase_offset = np.array([0.0, 0.0, 0.0])

        while True:
            for k in range(len(self.data)):
                tmp_val = self.data_config[k]
                sim_state = sim.get_state()
                sim_state.qpos[:] = tmp_val[:]
                sim_state.qpos[:3] +=  phase_offset[:]
                sim.set_state(sim_state)
                sim.forward()
                viewer.render()

            sim_state = sim.get_state()
            phase_offset = sim_state.qpos[:3]
            phase_offset[2] = 0

if __name__ == "__main__":
    test = MocapDM()
    curr_path = getcwd()
    test.play(curr_path + "/mujoco/motions/humanoid3d_spinkick.txt")