import numpy as np
import json
import os

naive_joint_mapping_humanoid_to_unitree_g1 = {
    "root": ("floating_base_joint", 0, np.array([0.85, 0.85, 0.85, 1., 1., 1., 1.])), # scale down xyz, because smaller robot needs feet to be at 0 and forward movement is smaller
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
    return qpos

def retarget_motion_humanoid_to_unitree_g1(motion):
    robot = "unitree_g1"
    from deepmimic_env import DPEnv, check_rewards_and_joint_limits
    humanoid_env = DPEnv(motion=motion)
    robot_env = DPEnv(motion=motion, load_mocap=False, robot=robot)
    # hqpos = humanoid_env.mocap.data_config
    robot_data_config = []
    dt = humanoid_env.mocap.dt
    for hqpos in humanoid_env.mocap.data_config: # for each frame
        g1qpos = np.array(robot_env.sim.data.qpos) * 0.0
        # naive mapping 
        for h_jname in humanoid_env.model.joint_names:
            g1joint_mapping = naive_joint_mapping_humanoid_to_unitree_g1[h_jname]
            if g1joint_mapping is None:
                continue
            g1_jname, offset, scale = g1joint_mapping
            if motion == "getup_facedown" and h_jname == "root":
                offset = np.array([0, 0, 0.17, 0, 0, 0, 0]) # move up, to avoid floor intersect
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
        # redo shoulders with euler angle conversion
        for side in ["left", "right"]:
            hr = g1qpos[humanoid_env.model.get_joint_qpos_addr(side + "_shoulder_x")]
            hp = g1qpos[humanoid_env.model.get_joint_qpos_addr(side + "_shoulder_y")]
            hy = g1qpos[humanoid_env.model.get_joint_qpos_addr(side + "_shoulder_z")]
            from transformations import euler_matrix, euler_from_matrix, quaternion_from_euler
            # humanoid is xy'z'' (intrinsic) and unitree is yx'z'' (intrinsic)
            mat = euler_matrix(hr, hp, hy, "rxyz")
            g1r, g1p, g1y = euler_from_matrix(mat, "ryxz")
            # we can't just use those angles because they don't respect limits. 
            # singularity smoothing:
            if True:
                tgt_quat = quaternion_from_euler(hr, hp, hy, "rxyz")
                VMX  = 15.
                joint_name = side + "_shoulder"
                # original values
                exo = g1r
                eyo = g1p
                ezo = g1y
                # create dicts if they don't exist
                if not "singularity_quat_error" in locals():
                    singularity_quat_error = {}
                    singularity_orig_config = {}
                    singularity_config = {}
                # get previous values
                if not joint_name + "_x" in singularity_quat_error:
                    exp = exo
                    eyp = eyo
                    ezp = ezo
                else:
                    exp = singularity_config[joint_name + "_x"][-1]
                    eyp = singularity_config[joint_name + "_y"][-1]
                    ezp = singularity_config[joint_name + "_z"][-1]
                # lims
                EX_LIM_MIN, EX_LIM_MAX = robot_env.model.jnt_range[robot_env.model.joint_name2id(joint_name + "_roll_joint")]
                EY_LIM_MIN, EY_LIM_MAX = robot_env.model.jnt_range[robot_env.model.joint_name2id(joint_name + "_pitch_joint")]
                EZ_LIM_MIN, EZ_LIM_MAX = robot_env.model.jnt_range[robot_env.model.joint_name2id(joint_name + "_yaw_joint")]
                ex_min = max(EX_LIM_MIN, exp - (VMX * dt))
                ex_max = min(EX_LIM_MAX, exp + (VMX * dt))
                ey_min = max(EY_LIM_MIN, eyp - (VMX * dt))
                ey_max = min(EY_LIM_MAX, eyp + (VMX * dt))
                ez_min = max(EZ_LIM_MIN, ezp - (VMX * dt))
                ez_max = min(EZ_LIM_MAX, ezp + (VMX * dt))
                ex_tgt = np.clip(exo, ex_min, ex_max)
                ey_tgt = np.clip(eyo, ey_min, ey_max)
                ez_tgt = np.clip(ezo, ez_min, ez_max)
                # if the desired values are within tolerance, just use them
                if np.allclose([exo, eyo, ezo], [ex_tgt, ey_tgt, ez_tgt]):
                    exn = exo
                    eyn = eyo
                    ezn = ezo
                    best_error = 0
                else:
                    best_error = np.inf
                    for exc in [ex_tgt, exp] + list(np.linspace(ex_min, ex_max, 6)):
                        for eyc in [ey_tgt, eyp] + list(np.linspace(ey_min, ey_max, 6)):
                            for ezc in [ez_tgt, ezp] + list(np.linspace(ez_min, ez_max, 6)):
                                quatn = quaternion_from_euler(exc, eyc, ezc, axes='rxyz')
                                error = min(np.linalg.norm(quatn - tgt_quat), np.linalg.norm(-quatn - tgt_quat))**2 # + 0.1 * np.linalg.norm([exc - exp, eyc - eyp, ezc - ezp])**2
                                if error < best_error:
                                    best_error = error
                                    exn = exc
                                    eyn = eyc
                                    ezn = ezc
                g1r, g1p, g1y = exn, eyn, ezn
                if motion == "getup_facedown":
                    g1p = g1p - 0.4 + hqpos[humanoid_env.model.get_joint_qpos_addr("chest_y")] # hack to make the motion more "natural" for the robot
                singularity_quat_error.setdefault(joint_name + "_x", []).append(best_error)
                singularity_quat_error.setdefault(joint_name + "_y", []).append(best_error)
                singularity_quat_error.setdefault(joint_name + "_z", []).append(best_error)
                singularity_orig_config.setdefault(joint_name + "_x", []).append(exo)
                singularity_orig_config.setdefault(joint_name + "_y", []).append(eyo)
                singularity_orig_config.setdefault(joint_name + "_z", []).append(ezo)
                singularity_config.setdefault(joint_name + "_x", []).append(exn)
                singularity_config.setdefault(joint_name + "_y", []).append(eyn)
                singularity_config.setdefault(joint_name + "_z", []).append(ezn)
                if len(singularity_config[joint_name + "_x"]) == len(humanoid_env.mocap.data_config) and side == "right":
                    # plot singularity fix
                    from matplotlib import pyplot as plt
                    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
                    for col, joint_name in enumerate(["left_shoulder", "right_shoulder"]):
                        for row, axis in enumerate(["x", "y", "z"]):
                            axs[col, row].plot(singularity_quat_error[joint_name + "_" + axis], label="quat_error")
                            axs[col, row].plot(singularity_orig_config[joint_name + "_" + axis], label="orig")
                            axs[col, row].plot(singularity_config[joint_name + "_" + axis], label="new")
                            axs[col, row].set_title(joint_name + "_" + axis)
                        EX_LIM_MIN, EX_LIM_MAX = robot_env.model.jnt_range[robot_env.model.joint_name2id(joint_name + "_roll_joint")]
                        EY_LIM_MIN, EY_LIM_MAX = robot_env.model.jnt_range[robot_env.model.joint_name2id(joint_name + "_pitch_joint")]
                        EZ_LIM_MIN, EZ_LIM_MAX = robot_env.model.jnt_range[robot_env.model.joint_name2id(joint_name + "_yaw_joint")]
                        axs[col, 0].axhline(EX_LIM_MAX, color='r', linestyle='--')
                        axs[col, 0].axhline(EX_LIM_MIN, color='r', linestyle='--')
                        axs[col, 1].axhline(EY_LIM_MAX, color='r', linestyle='--')
                        axs[col, 1].axhline(EY_LIM_MIN, color='r', linestyle='--')
                        axs[col, 2].axhline(EZ_LIM_MAX, color='r', linestyle='--')
                        axs[col, 2].axhline(EZ_LIM_MIN, color='r', linestyle='--')
                    plt.legend()
                    plt.show()
            # assign
            g1qpos[robot_env.model.get_joint_qpos_addr(side + "_shoulder_roll_joint")] = g1r
            g1qpos[robot_env.model.get_joint_qpos_addr(side + "_shoulder_pitch_joint")] = g1p
            g1qpos[robot_env.model.get_joint_qpos_addr(side + "_shoulder_yaw_joint")] = g1y
        robot_data_config.append([dt] + g1qpos.tolist())
        print("Frame", len(robot_data_config), "of", len(humanoid_env.mocap.data_config))
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
    retarget_motion_humanoid_to_unitree_g1("getup_facedown")
