#  帧堆叠技术
import time
import gymnasium as gym
import json
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, DQN, DDPG, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from tqdm import tqdm  # 引入 tqdm 用于显示进度条
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from production.envs.transport import Transport
from production.envs.production_env import ProductionEnv  # 引入自定义的ProductionEnv
from production.envs.frame_stacking_env import FrameStackingWrapper, StateNormalizingFrameStack
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# 加载帧堆叠配置
with open('config/ppo_framestack.json', 'r') as f:
    config = json.load(f)

# 提取帧堆叠配置
frame_stack_config = config.pop('frame_stack_config', {})

# 设置训练参数
timesteps = 180  # 每个回合的时间步
episodes = 1000  # 总回合数
total_timesteps = timesteps * episodes

# 创建基础环境
base_env = ProductionEnv(max_episode_timesteps=timesteps)

# 应用帧堆叠包装器
if frame_stack_config.get('normalize', False):
    env = StateNormalizingFrameStack(
        base_env, # 基础环境
        num_stack=frame_stack_config.get('num_stack', 4), # 将状态向量堆叠成4帧 
        normalize=True, # 归一化
        running_mean_window=frame_stack_config.get('running_mean_window', 1000) # 归一化窗口
    )
else:
    env = FrameStackingWrapper(
        base_env,
        num_stack=frame_stack_config.get('num_stack', 4),
        skip_frames=frame_stack_config.get('skip_frames', 1),
        channel_order=frame_stack_config.get('channel_order', 'last')
    )

check_env(env)

algorithm = "PPO"
policy = config.pop('policy')

# 创建模型 - 使用标准MLP策略
if algorithm == "PPO":
    model = PPO(policy, env, **config)
elif algorithm == "DQN":
    model = DQN(policy, env, **config)
elif algorithm == "A2C":
    model = A2C(policy, env, **config)
elif algorithm == "SAC":
    model = SAC(policy, env, **config)
elif algorithm == "DDPG":
    model = DDPG(policy, env, **config)
elif algorithm == "TD3":
    model = TD3(policy, env, **config)

# 创建一个空列表来保存每个 episode 的奖励
rewards = []

# 实时绘图设置
plt.ion()  # 开启matplotlib交互模式
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('帧堆叠网络训练：奖励变化曲线')

# 自定义回调函数
class FrameStackRewardCallback(BaseCallback):
    def __init__(self, total_episodes, max_timesteps_per_episode, verbose=0):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.current_episode_reward = 0
        self.timesteps_in_episode = 0
        self.episode_rewards = []
        self.losses = []
        self.pbar = tqdm(total=self.total_episodes, desc="帧堆叠训练进度", unit="episode")

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.timesteps_in_episode += 1

        if self.timesteps_in_episode >= self.max_timesteps_per_episode:
            avg_reward = self.current_episode_reward / self.max_timesteps_per_episode
            self.episode_rewards.append(avg_reward)
            
            # 保存奖励数据
            with open("rewards_FrameStack.txt", "a") as f:
                f.write(f"{avg_reward}\n")

            # Reset counters
            self.current_episode_reward = 0
            self.timesteps_in_episode = 0

            # Update reward plot
            ax.clear()
            ax.set_xlabel('Episode')
            ax.set_ylabel('平均奖励')
            ax.set_title(f'帧堆叠网络训练：奖励变化曲线 (当前: {len(self.episode_rewards)} episodes)')
            ax.grid(True, alpha=0.3)
            
            episodes_range = range(1, len(self.episode_rewards) + 1)
            ax.plot(episodes_range, self.episode_rewards, 'g-', alpha=0.7, linewidth=1.5, label='帧堆叠奖励')
            
            # 添加移动平均线
            if len(self.episode_rewards) > 10:
                window_size = min(20, len(self.episode_rewards) // 2)
                if window_size > 1:
                    moving_avg = []
                    for i in range(window_size-1, len(self.episode_rewards)):
                        avg = np.mean(self.episode_rewards[i-window_size+1:i+1])
                        moving_avg.append(avg)
                    
                    ax.plot(range(window_size, len(self.episode_rewards) + 1), moving_avg, 
                            'r-', linewidth=2, label=f'移动平均({window_size})')
            
            # 标注最新数值
            if self.episode_rewards:
                latest_reward = self.episode_rewards[-1]
                ax.annotate(f'{latest_reward:.3f}', 
                           xy=(len(self.episode_rewards), latest_reward),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.legend()
            plt.draw()
            plt.pause(0.01)
            self.pbar.update(1)
            
            # 每50个episode输出一次统计信息
            if len(self.episode_rewards) % 50 == 0:
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else avg_reward
                print(f"\n回合 {len(self.episode_rewards)}: 当前奖励={avg_reward:.4f}, 最近10回合均值={recent_avg:.4f}")

        return True

    def _on_rollout_end(self) -> None:
        # 记录训练损失
        logger_data = self.model.logger.name_to_value
        loss = logger_data.get("train/value_loss")
        if loss is not None:
            self.losses.append(loss)
            with open("losses_FrameStack.txt", "a") as f:
                f.write(f"{loss}\n")

    def _on_training_end(self) -> None:
        self.pbar.close()
        print("\n帧堆叠训练完成！")
        
        # 保存最终的训练曲线
        plt.figure(figsize=(15, 5))
        
        # 奖励曲线
        plt.subplot(1, 3, 1)
        if self.episode_rewards:
            plt.plot(self.episode_rewards, 'g-', alpha=0.7, label='帧堆叠奖励')
            if len(self.episode_rewards) > 20:
                window_size = 20
                moving_avg = [np.mean(self.episode_rewards[i:i+window_size]) 
                            for i in range(len(self.episode_rewards)-window_size+1)]
                plt.plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label='移动平均')
        plt.xlabel('Episode')
        plt.ylabel('平均奖励')
        plt.title('帧堆叠训练：奖励变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失曲线
        plt.subplot(1, 3, 2)
        if self.losses:
            plt.plot(self.losses, 'orange', alpha=0.8, label='训练损失')
        plt.xlabel('更新轮次')
        plt.ylabel('损失')
        plt.title('帧堆叠训练：损失变化')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 训练统计
        plt.subplot(1, 3, 3)
        if self.episode_rewards:
            stats_text = f"""训练统计:

算法: 帧堆叠MLP
总回合数: {len(self.episode_rewards)}
平均奖励: {np.mean(self.episode_rewards):.4f}
最高奖励: {np.max(self.episode_rewards):.4f}
最低奖励: {np.min(self.episode_rewards):.4f}
标准差: {np.std(self.episode_rewards):.4f}

最近10回合:
平均奖励: {np.mean(self.episode_rewards[-10:]):.4f}

帧堆叠设置:
堆叠数: {frame_stack_config.get('num_stack', 4)}
跳帧: {frame_stack_config.get('skip_frames', 1)}
归一化: {frame_stack_config.get('normalize', False)}
"""
            plt.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("framestack_training_curves.png", dpi=300, bbox_inches='tight')
        print("训练曲线已保存为 framestack_training_curves.png")

# 创建回调
callback = FrameStackRewardCallback(total_episodes=episodes, max_timesteps_per_episode=timesteps)

print("开始帧堆叠网络训练...")
print(f"网络架构：")
print(f"- 特征提取器：帧堆叠MLP")
print(f"- 原始状态维度：{base_env.observation_space.shape}")
print(f"- 堆叠后维度：{env.observation_space.shape}")
print(f"- 帧堆叠配置：{frame_stack_config}")

# 开始训练
start_time = time.time()
model.learn(total_timesteps=total_timesteps, callback=callback)
end_time = time.time()

print(f"\n训练完成！总用时：{end_time - start_time:.2f}秒")

# 保存模型
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_path = f"model_history2/framestack_model_{timestamp}.zip"
model.save(model_path)
print(f"帧堆叠模型已保存到：{model_path}")

# 保存环境统计数据
env.unwrapped.statistics.update({'time_end': env.unwrapped.env.now})

# 保持图像显示
plt.ioff()
plt.show() 
