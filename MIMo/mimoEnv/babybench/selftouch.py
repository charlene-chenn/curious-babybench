""" 

"""
import os
import numpy as np
import mujoco

from mimoEnv.babybench.base import BabyBenchEnv, DEFAULT_SIZE, SCENE_XML
from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY
from mimoActuation.actuation import SpringDamperModel
import mimoEnv.utils as env_utils
import mimoEnv.babybench.utils as bb_utils

class BabyBenchSelfTouchEnv(BabyBenchEnv):
    """ 
    
    """
    def __init__(self,
                 model_path=SCENE_XML,
                 frame_skip=1,
                 width=DEFAULT_SIZE,
                 height=DEFAULT_SIZE,
                 **kwargs):

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         width=width,
                         height=height,
                         **kwargs)

        # initialize behavior-relevant variables
        self.right_hand_geoms = env_utils.get_geoms_for_body(self.model, env_utils.get_body_id(self.model, body_name="right_hand"))
        self.left_hand_geoms = env_utils.get_geoms_for_body(self.model, env_utils.get_body_id(self.model, body_name="left_hand"))
        self.mimo_bodies = env_utils.get_child_bodies(self.model, env_utils.get_body_id(self.model, body_name="hip"))
        self.mimo_geoms = np.concatenate([np.array(env_utils.get_geoms_for_body(self.model, body_id)) for body_id in self.mimo_bodies])
        
    def _info(self):
        # Get all contacts
        contacts = self.data.contact
        for idx in range(len(contacts.geom1)):
            # Check if hand in contact
            if contacts.geom1[idx] in self.right_hand_geoms:
                other_geom = contacts.geom2[idx]
                # Check if other geom is in mimo's body and is new contact
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_right_hand):
                    # Add to list of contacts
                    self._self_touch_right_hand.append(int(other_geom))
            elif contacts.geom2[idx] in self.right_hand_geoms:
                other_geom = contacts.geom1[idx]
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_right_hand):
                    self._self_touch_right_hand.append(int(other_geom))
            elif contacts.geom1[idx] in self.left_hand_geoms:
                other_geom = contacts.geom2[idx]
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_left_hand):
                    self._self_touch_left_hand.append(int(other_geom))
            elif contacts.geom2[idx] in self.left_hand_geoms:
                other_geom = contacts.geom1[idx]
                if (other_geom in self.mimo_geoms) and (other_geom not in self._self_touch_left_hand):
                    self._self_touch_left_hand.append(int(other_geom))
        info = {
            'right_hand_touches' : self._self_touch_right_hand,
            'left_hand_touches'  : self._self_touch_left_hand,
        }
        return info

    def _info_init(self):
        self._self_touch_right_hand = []
        self._self_touch_left_hand = []

    def _randomize_reset(self):
        # perform 10 steps with random actions
        for _ in range(10):
            self._set_action(self.action_space.sample())
            self._single_mujoco_step()
    