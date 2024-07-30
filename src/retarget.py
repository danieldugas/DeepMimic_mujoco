import numpy as np
import json
import os

naive_joint_mapping_humanoid_to_unitree_g1 = {
    "root": ("floating_base_joint", 0, +1.0),
    "chest_x": None,
    "chest_y": None,
    "chest_z": ("torso_joint", 0, +1.0),
    "neck_x": None,
    "neck_y": None,
    "neck_z": None,
    "right_shoulder_x": ("right_shoulder_roll_joint", 0, +1.0),
    "right_shoulder_y": ("right_shoulder_pitch_joint", 0, +1.0),
    "right_shoulder_z": ("right_shoulder_yaw_joint", 0, +1.0),
    "right_elbow": ("right_elbow_pitch_joint", 1.57, -1.0),
    # None -> "right_elbow_roll_joint",
    "left_shoulder_x": ("left_shoulder_roll_joint", 0, +1.0),
    "left_shoulder_y": ("left_shoulder_pitch_joint", 0, +1.0),
    "left_shoulder_z": ("left_shoulder_yaw_joint", 0, +1.0),
    "left_elbow": ("left_elbow_pitch_joint", 1.57, -1.0),
    # None -> "left_elbow_roll_joint",
    "right_hip_x": ("right_hip_roll_joint", 0, +1.0),
    "right_hip_y": ("right_hip_pitch_joint", 0, +1.0),
    "right_hip_z": ("right_hip_yaw_joint", 0, +1.0),
    "right_knee": ("right_knee_joint", 0, -1.0),
    "right_ankle_x": ("right_ankle_roll_joint", 0, +1.0),
    "right_ankle_y": ("right_ankle_pitch_joint", 0, +1.0),
    "right_ankle_z": None,
    "left_hip_x": ("left_hip_roll_joint", 0, +1.0),
    "left_hip_y": ("left_hip_pitch_joint", 0, +1.0),
    "left_hip_z": ("left_hip_yaw_joint", 0, +1.0),
    "left_knee": ("left_knee_joint", 0, -1.0),
    "left_ankle_x": ("left_ankle_roll_joint", 0, +1.0),
    "left_ankle_y": ("left_ankle_pitch_joint", 0, +1.0),
    "left_ankle_z": None
}

def retarget_joints_humanoid_to_unitree_g1(qpos):

    return qpos2

def retarget_motion_humanoid_to_unitree_g1(motion):
    robot = "unitree_g1"
    from deepmimic_env import DPEnv, check_rewards_and_joint_limits
    humanoid_env = DPEnv(motion=motion)
    robot_env = DPEnv(motion=motion, load_mocap=False, robot=robot)
    # hqpos = humanoid_env.mocap.data_config
    robot_data_config = []
    dt = humanoid_env.mocap.dt
    for hqpos in humanoid_env.mocap.data_config:
        g1qpos = np.array(robot_env.sim.data.qpos) * 0.0
        for h_jname in humanoid_env.model.joint_names:
            g1joint_mapping = naive_joint_mapping_humanoid_to_unitree_g1[h_jname]
            if g1joint_mapping is None:
                continue
            g1_jname, offset, scale = g1joint_mapping
            g1_jaddr = robot_env.model.get_joint_qpos_addr(g1_jname)
            if isinstance(g1_jaddr, tuple):
                g1_jstart, g1_jend = g1_jaddr
            else:
                g1_jstart, g1_jend = g1_jaddr, g1_jaddr + 1
            h_jaddr = humanoid_env.model.get_joint_qpos_addr(h_jname)
            if isinstance(h_jaddr, tuple):
                h_jstart, h_jend = h_jaddr
            else:
                h_jstart, h_jend = h_jaddr, h_jaddr + 1
            g1qpos[g1_jstart:g1_jend] = hqpos[h_jstart:h_jend] * scale + offset
        robot_data_config.append([dt] + g1qpos.tolist())
    # to list of lists, json
    json_dict = {
        "Format": "direct_qpos",
        "JointNames": robot_env.model.joint_names,
        "Loop": humanoid_env.mocap.loop,
        "Frames": robot_data_config,
    }
    mocap_path = robot_env.config.mocap_path
    if os.path.exists(mocap_path):
        raise Exception("File exists: %s"%(mocap_path))
        # print("File exists. refusing to overwrite")
    else:
        with open(mocap_path, "w") as f:
            json.dump(json_dict, f, indent=4)
        print("Retargeted motion saved to", mocap_path)
    
    check_rewards_and_joint_limits(motion=motion, robot=robot)


    

if __name__ == "__main__":
    retarget_motion_humanoid_to_unitree_g1("run")
