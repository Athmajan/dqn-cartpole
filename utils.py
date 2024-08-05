import cv2
import numpy as np
from random import randint
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

        self.buffer = np.zeros((num_stack,h,w),'uint8')
        self.frame = None

    def _preprocess_frame(self,frame):
        image = cv2.resize(frame,(self.w,self.h))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return image

    def render(self,mode):
        self.env.render(mode)
    
    def reset(self):
        im,_ = self.env.reset()
        # self.env.step(1) should we do this at reset?
        # self.env.step(1)
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer = np.stack([im]*self.n,0)
        return self.buffer.copy()
    
    def step(self, action):
        im, reward, terminated, truncated, info = self.env.step(action)
        self.frame = im.copy()
        done = terminated or truncated
        im = self._preprocess_frame(im)
        # if we have four frames stacked
        # we wanna take frame 0 and set it in 1 position
        # frame 1 set at 2 position
        # frame 2 set at 3 position
        # frame 3 set at 4 position
        # frame 4 goes away
        self.buffer[1:self.n,:,:] = self.buffer[0:self.n-1,:,:]
        self.buffer[0,:,:] = im
        return self.buffer.copy(), reward, done, info
    
    def render(self,mode):
        if mode == "rbg_array":
            return self.frame
        return super(FrameStackingAndResizingEnv,self.render(mode))

    @property
    def observation_space(self):
        # clean this up later with gym.spaces.Box()
        return np.zeros((self.n,self.h,self.w))
    
    @property
    def action_space(self):
        return self.env.action_space

if __name__ == "__main__":
    import gym
    np.bool8 = np.bool

    env = gym.make("Breakout-v4")
    env = FrameStackingAndResizingEnv(env, 480,640)
    im = env.reset()
    idx = 0
    ims = []
    for i in range(im.shape[-1]):
        ims.append(im[:,:,i])
    cv2.imwrite(f"tmp/{idx}.jpg",np.hstack(ims))

    env.step(1)

    for _ in range(10):
        idx += 1
        # import ipdb; ipdb.set_trace()
        im, reward,done, info = env.step(randint(0,3))
        ims = []
        for i in range(im.shape[-1]):
            ims.append(im[:,:,i])
        cv2.imwrite(f"tmp/{idx}.jpg",np.hstack(ims))






