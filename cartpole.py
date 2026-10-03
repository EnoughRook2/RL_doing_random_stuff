import gymnasium as gym
import random

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env , epsilon: float|int=0.1):
        super(RandomActionWrapper,self).__init__(env)
        self.epsilon=epsilon
    def action(self, action: gym.core.WrapperActType)->gym.core.WrapperActType:
        if random.random()<self.epsilon:
            action=self.env.action_space.sample()
            print(f"Random action:{action}")
            return action
        return action


if __name__=="__main__":
    env=gym.make("CartPole-v1",render_mode="rgb_array")
    env=RandomActionWrapper(env)
    env = gym.wrappers.HumanRendering(env)
    # env = gym.wrappers.RecordVideo(env, video_folder="video") (this is for recording)
    total_reward=0.0
    total_steps=0
    obs,_=env.reset()
    while True:
        action=env.action_space.sample()
        obs,reward,is_done,is_trunc,_=env.step(action)
        total_reward+=reward
        total_steps +=1
        if is_done:
            break
    print("Episode done in %d steps, total reward %.4f"%(total_steps,total_reward))
    env.close()
