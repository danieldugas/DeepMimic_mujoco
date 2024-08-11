import os

class Config(object):
    def __init__(self, motion=None, robot="humanoid3d"):
        self.all_motions = ['backflip', 'cartwheel', 'crawl', 'dance_a', 'dance_b', 'getup_facedown'
                    'getup_faceup', 'jump', 'kick', 'punch', 'roll', 'run', 'spin', 'spinkick',
                    'walk']
        self.acyclical_motions = ["getup_faceup", "getup_facedown", "getup_facedown_slow", "getup_facedown_slow_FSI"]
        self.floor_motions = ["getup_faceup", "getup_facedown", "getup_facedown_slow", "getup_facedown_slow_FSI"]
        self.curr_path = os.path.expanduser("~/Code/DeepMimic_mujoco/src")
        self.motion = 'walk' if motion is None else motion
        self.robot = robot
        self.env_name = "deepmimic_" + self.robot

        self.motion_folder = '/mujoco/motions'
        self.xml_folder = '/mujoco/humanoid_deepmimic/envs/asset'
        self.xml_test_folder = '/mujoco_test/'

        self.mocap_path = "%s%s/%s_%s.txt"%(self.curr_path, self.motion_folder, self.robot, self.motion)
        self.xml_path = "%s%s/%s.xml"%(self.curr_path, self.xml_folder, self.env_name)
        self.xml_path_test = "%s%s/%s_test.xml"%(self.curr_path, self.xml_test_folder, self.env_name)

        if self.robot == "humanoid3d":
            self.torso_body_name = "chest"
            self.lfoot_geom_name = "left_ankle"
            self.rfoot_geom_name = "right_ankle"
            self.floor_geom_name = "floor"
            self.endeffector_geom_names = ["left_ankle", "right_ankle", "left_wrist", "right_wrist"]
            self.low_z = 0.7
        elif self.robot == "unitree_g1":
            self.torso_body_name = "pelvis"
            self.lfoot_geom_name = "left_foot"
            self.rfoot_geom_name = "right_foot"
            self.floor_geom_name = "floor"
            self.endeffector_geom_names = ["left_foot", "right_foot", "left_hand", "right_hand"]
            self.low_z = 0.4
        else:
            raise Exception("Unknown robot: %s"%(self.robot))
