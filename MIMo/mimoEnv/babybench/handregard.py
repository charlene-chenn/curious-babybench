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

class BabyBenchHandRegardEnv(BabyBenchEnv):
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
        
    def _info(self):
        # Get positions of hands and eyes
        right_hand_pos = self.data.body('right_hand').xpos
        left_hand_pos = self.data.body('left_hand').xpos
        right_eye_pos = self.data.body('right_eye').xpos
        left_eye_pos = self.data.body('left_eye').xpos
        # Compute vectors from eyes to hands
        right_eye_to_right_hand = right_hand_pos - right_eye_pos
        right_eye_to_left_hand = left_hand_pos - right_eye_pos
        left_eye_to_right_hand = right_hand_pos - left_eye_pos
        left_eye_to_left_hand = left_hand_pos - left_eye_pos
        # Get rotation matrices of eyes
        right_eye_rot = self.data.body('right_eye').xmat.reshape(3, 3)
        left_eye_rot = self.data.body('left_eye').xmat.reshape(3, 3)
        # Camera forward direction is the negative z-axis in the eye's local frame
        right_eye_forward = -right_eye_rot[:, 2]
        left_eye_forward = -left_eye_rot[:, 2]
        # Compute angles between eye forward directions and eyes-to-hands vectors
        right_eye_right_hand_angle = bb_utils.angle_between_vectors(right_eye_forward, right_eye_to_right_hand)
        right_eye_left_hand_angle = bb_utils.angle_between_vectors(right_eye_forward, right_eye_to_left_hand)
        left_eye_right_hand_angle = bb_utils.angle_between_vectors(left_eye_forward, left_eye_to_right_hand)
        left_eye_left_hand_angle = bb_utils.angle_between_vectors(left_eye_forward, left_eye_to_left_hand)
        # Field of view of eye cameras
        fovy = 30.0
        # Add to hand regard times
        self._hand_regard_right_eye_right_hand += int(right_eye_right_hand_angle < fovy/2)
        self._hand_regard_right_eye_left_hand += int(right_eye_left_hand_angle < fovy/2)
        self._hand_regard_left_eye_right_hand += int(left_eye_right_hand_angle < fovy/2)
        self._hand_regard_left_eye_left_hand += int(left_eye_left_hand_angle < fovy/2)
        info = {'right_eye_right_hand': self._hand_regard_right_eye_right_hand,
                'right_eye_left_hand': self._hand_regard_right_eye_left_hand,
                'left_eye_right_hand': self._hand_regard_left_eye_right_hand,
                'left_eye_left_hand': self._hand_regard_left_eye_left_hand}
        return info

    def _info_init(self):
        self._hand_regard_right_eye_right_hand = 0
        self._hand_regard_right_eye_left_hand = 0
        self._hand_regard_left_eye_right_hand = 0
        self._hand_regard_left_eye_left_hand = 0

    def _randomize_reset(self):
        # perform 10 steps with random actions
        for _ in range(10):
            self._set_action(self.action_space.sample())
            self._single_mujoco_step()
    