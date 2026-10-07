import typing as tt
import gymnasium as gym
from collections import defaultdict,Counter
from torch.utils.tensorboard.writer import SummaryWriter
from gymnasium.wrappers import RecordVideo

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

State= int
Action=int
RewardKey=tt.Tuple[State,Action,State]
TransitKey=tt.Tuple[State,Action]

class Agent:
    def __init__(self):
        self.env=gym.make(ENV_NAME)
        self.state,_=self.env.reset()
        self.rewards: tt.Dict[RewardKey,float|int]=defaultdict(float)
        self.transits:tt.Dict[TransitKey,Counter]=defaultdict(Counter)
        self.values: tt.Dict[TransitKey, float|int] = defaultdict(float)

    def play_n_random_steps(self,n:int):
        for _ in range(n):
            action=self.env.action_space.sample()
            new_state,reward,is_done,is_trunc,_=self.env.step(action)
            rw_key=(self.state,action,new_state)
            self.rewards[rw_key]=float(reward)
            tr_key=(self.state,action)
            self.transits[tr_key][new_state] +=1
            if is_done or is_trunc:
                self.state,_= self.env.reset()
            else:
                self.state=new_state

    def select_action(self, state: State) -> Action:
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env:gym.Env)->float|int:
        total_reward=0.0
        state,_= env.reset()
        while True:
            action=self.select_action(state)
            new_state,reward,is_done,is_trunc,_=env.step(action)
            rw_key=(state,action,new_state)
            self.rewards[rw_key]=float(reward)
            tr_key=(state,action)
            self.transits[tr_key][new_state] +=1
            total_reward +=reward
            if is_done or is_trunc:
                break
            state=new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    rw_key = (state, action, tgt_state)
                    reward = self.rewards[rw_key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


if __name__=="__main__":
    test_env=gym.make(ENV_NAME)
    agent=Agent()
    writer=SummaryWriter(comment="frozenlake-q-iteration")
    iteration=0
    best_reward=0.0
    while True:
        iteration +=1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward=0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /=TEST_EPISODES
        writer.add_scalar("reward",reward,iteration)
        if reward > best_reward:
            print(f"{iteration}:Best reward updated {best_reward:.3} --> {reward:.3}")
            best_reward=reward
        if reward >0.95:
            print("solved in %d iterations" % iteration)
            break
    print("Finished")
    writer.close()
    test_env.close()
    agent.env.close()

    print("Recording solved video...")
    demo_env = gym.make(ENV_NAME, render_mode="rgb_array")
    demo_env = RecordVideo(demo_env,video_folder="video/frozenlake",name_prefix="frozenlake-q-iteration")
    agent.play_episode(demo_env)
    demo_env.close()
    print("Finished!")
