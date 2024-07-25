import os

class Config(object):
    def __init__(self, motion=None):
        self.all_motions = ['backflip', 'cartwheel', 'crawl', 'dance_a', 'dance_b', 'getup_facedown'
                    'getup_faceup', 'jump', 'kick', 'punch', 'roll', 'run', 'spin', 'spinkick',
                    'walk']
        self.acyclical_motions = ['getup_faceup', 'getup_facedown']
        self.floor_motions = ["getup_faceup", "getup_facedown"]
        self.curr_path = os.path.expanduser("~/Code/DeepMimic_mujoco/src")
        self.motion = 'walk' if motion is None else motion
        self.env_name = "dp_env_v3"

        self.motion_folder = '/mujoco/motions'
        self.xml_folder = '/mujoco/humanoid_deepmimic/envs/asset'
        self.xml_test_folder = '/mujoco_test/'

        self.mocap_path = "%s%s/humanoid3d_%s.txt"%(self.curr_path, self.motion_folder, self.motion)
        self.xml_path = "%s%s/%s.xml"%(self.curr_path, self.xml_folder, self.env_name)
        self.xml_path_test = "%s%s/%s_test.xml"%(self.curr_path, self.xml_test_folder, self.env_name)
