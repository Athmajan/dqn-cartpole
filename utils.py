import cv2
import numpy as np
'''
env.reset() - > obs : ndarray
env.step(action : int) -> obs : ndarray, rew: float, info:Dict
env.render()

0th frame is latest
1st frame is the one before etc.
'''


class FrameStackingAndResizingEnv:
    def __init__(self,env,w,h,num_stack=4):
        self.env = env
        self.n = num_stack
        self.w = w
        self.h = h

        self.buffer = np.zeros((h,w,num_stack),'uint8')

    def _preprocess_frame(self,frame):
        image = cv2.resize(frame,(self.w,self.h))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return image

    def render(self,mode):
        self.env.render(mode)
    
    def reset(self):
        im = self.env.reset()
        im = self._preprocess_frame(im)
        self.buffer[:,:,0] = np.dstack([im]*self.n)
        return self.buffer.copy()
