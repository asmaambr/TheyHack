import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import os
import random
from collections import deque

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class ServerRoomEnv(gym.Env):
    """
    Custom Environment that simulates a server room with temperature and humidity sensors.
    Actions include turning fans on/off and controlling fan speed.
    """
    def __init__(self):
        super(ServerRoomEnv, self).__init__()

        # Define action space: [fan_on (0/1), fan_speed (0-1)]
        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        # Define observation space: [temperature, humidity]
        self.observation_space = spaces.Box(low=np.array([15, 10]),
                                           high=np.array([40, 90]),
                                           dtype=np.float32)

        # Environment parameters
        self.optimal_temp_range = (20, 25)  # Optimal temperature range in Celsius
        self.optimal_humidity_range = (40, 60)  # Optimal humidity range in percentage
        self.critical_temp = 30  # Critical temperature threshold
        self.critical_humidity = 70  # Critical humidity threshold

        # Internal state
        self.temperature = 22.0
        self.humidity = 50.0
        self.time_step = 0
        self.max_steps = 1000

        # Server heat generation parameters
        self.base_heat_rate = 0.1  # Base temperature increase per step
        self.server_load = 0.5  # Server load (0-1)

        # External factors (can be extended with actual data)
        self.external_temp = 25.0
        self.external_humidity = 55.0

    def step(self, action):
        # Unpack action
        fan_on = action[0] > 0.5  # Binary threshold
        fan_speed = action[1]  # Continuous between 0-1

        # Calculate effective cooling based on fan state and speed
        effective_cooling = 0
        if fan_on:
            effective_cooling = 0.2 * fan_speed

        # Calculate heat generation based on server load
        heat_generation = self.base_heat_rate * (0.5 + self.server_load)

        # Update temperature based on cooling and heat generation
        delta_temp = heat_generation - effective_cooling

        # External temperature influence
        external_influence = 0.05 * (self.external_temp - self.temperature)

        # Update temperature
        self.temperature += delta_temp + external_influence

        # Update humidity (simplified model)
        if fan_on:
            # Fans tend to reduce humidity
            humidity_change = -0.2 * fan_speed + 0.1 * np.random.randn()
        else:
            # Humidity tends to increase when fans are off
            humidity_change = 0.1 + 0.05 * np.random.randn()

        # External humidity influence
        external_humidity_influence = 0.05 * (self.external_humidity - self.humidity)

        self.humidity += humidity_change + external_humidity_influence
        self.humidity = np.clip(self.humidity, 10, 90)

        # Get current state
        state = np.array([self.temperature, self.humidity], dtype=np.float32)

        # Calculate reward
        reward = self._calculate_reward(fan_on, fan_speed)

        # Increment time step
        self.time_step += 1

        # Check if episode is done
        done = (self.time_step >= self.max_steps or
                self.temperature >= self.critical_temp or
                self.humidity >= self.critical_humidity)

        # Additional info
        info = {
            'temp_status': 'optimal' if self.optimal_temp_range[0] <= self.temperature <= self.optimal_temp_range[1] else 'suboptimal',
            'humidity_status': 'optimal' if self.optimal_humidity_range[0] <= self.humidity <= self.optimal_humidity_range[1] else 'suboptimal',
            'power_consumption': fan_speed if fan_on else 0
        }

        return state, reward, done, info

    def _calculate_reward(self, fan_on, fan_speed):
        # Initialize reward
        reward = 0

        # Reward for maintaining optimal temperature
        if self.optimal_temp_range[0] <= self.temperature <= self.optimal_temp_range[1]:
            reward += 2.0
        else:
            # Penalty increases as temperature moves away from optimal range
            temp_deviation = min(
                abs(self.temperature - self.optimal_temp_range[0]),
                abs(self.temperature - self.optimal_temp_range[1])
            )
            reward -= 0.5 * temp_deviation

        # Reward for maintaining optimal humidity
        if self.optimal_humidity_range[0] <= self.humidity <= self.optimal_humidity_range[1]:
            reward += 1.0
        else:
            # Penalty for humidity out of range
            humidity_deviation = min(
                abs(self.humidity - self.optimal_humidity_range[0]),
                abs(self.humidity - self.optimal_humidity_range[1])
            )
            reward -= 0.3 * humidity_deviation

        # Energy consumption penalty
        if fan_on:
            energy_penalty = 0.1 + 0.4 * fan_speed
            reward -= energy_penalty

        # Critical condition penalty
        if self.temperature >= self.critical_temp:
            reward -= 10.0
        if self.humidity >= self.critical_humidity:
            reward -= 5.0

        return reward

    def reset(self):
        # Reset environment to initial state with some randomness
        self.temperature = 22.0 + np.random.uniform(-2, 2)
        self.humidity = 50.0 + np.random.uniform(-5, 5)
        self.time_step = 0

        # Randomize server load for each episode
        self.server_load = 0.3 + 0.6 * np.random.random()

        # Randomize external conditions
        self.external_temp = 25.0 + np.random.uniform(-5, 5)
        self.external_humidity = 55.0 + np.random.uniform(-10, 10)

        return np.array([self.temperature, self.humidity], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.time_step}")
        print(f"Temperature: {self.temperature:.2f}°C")
        print(f"Humidity: {self.humidity:.2f}%")
        print(f"Server Load: {self.server_load:.2f}")
        print(f"External: {self.external_temp:.2f}°C, {self.external_humidity:.2f}%")
        print("-" * 30)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            tf.convert_to_tensor(self.state[ind]),
            tf.convert_to_tensor(self.action[ind]),
            tf.convert_to_tensor(self.next_state[ind]),
            tf.convert_to_tensor(self.reward[ind]),
            tf.convert_to_tensor(self.done[ind])
        )


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.l1 = layers.Dense(256, activation='relu')
        self.l2 = layers.Dense(256, activation='relu')
        self.mean = layers.Dense(action_dim)
        self.log_std = layers.Dense(action_dim)

    def call(self, state):
        a = self.l1(state)
        a = self.l2(a)

        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)

        return mean, std

    def sample(self, state):
        mean, std = self.call(state)

        # Using reparameterization trick
        normal = tf.random.normal(shape=mean.shape)
        z = mean + std * normal

        # Apply tanh to bound actions
        action = tf.tanh(z)

        # Scale to action range
        action = action * self.max_action

        # Log probability
        log_prob = self._log_prob(mean, std, action)

        return action, log_prob

    def _log_prob(self, mean, std, action):
        # Reparameterization trick
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((action - mean) / std), axis=1
        )
        log_prob -= 0.5 * tf.reduce_sum(tf.math.log(2 * np.pi * tf.square(std)), axis=1)

        return log_prob


class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.l1 = layers.Dense(256, activation='relu')
        self.l2 = layers.Dense(256, activation='relu')
        self.value = layers.Dense(1)

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.value(x)


class SAC:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.critic_1 = Critic(state_dim)
        self.critic_1_target = Critic(state_dim)
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_1_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.critic_2 = Critic(state_dim)
        self.critic_2_target = Critic(state_dim)
        self.critic_2_target.set_weights(self.critic_2.get_weights())
        self.critic_2_optimizer = tf.keras.optimizers.Adam(3e-4)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

        # SAC specific parameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha  # entropy regularization coefficient

        # Initialize target networks with actor weights
        self.update_target_networks(tau=1.0)

    def select_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        if evaluate:
            mean, _ = self.actor(state)
            return mean[0].numpy() * self.max_action
        else:
            action, _ = self.actor.sample(state)
            return action[0].numpy()

    def train(self, batch_size=64):
        if self.replay_buffer.size < batch_size:
            return

        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)

        # Get next actions and their log probs for entropy regularization
        next_action, next_log_prob = self.actor.sample(next_state)

        # Update critics
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Target Q-values
            target_q1 = self.critic_1_target(next_state)
            target_q2 = self.critic_2_target(next_state)
            target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.discount * target_q

            # Current Q-values
            current_q1 = self.critic_1(state)
            current_q2 = self.critic_2(state)

            # Compute critic losses
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))

        # Optimize critics
        critic1_grad = tape1.gradient(critic1_loss, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic1_grad, self.critic_1.trainable_variables))

        critic2_grad = tape2.gradient(critic2_loss, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(zip(critic2_grad, self.critic_2.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            # Sample actions from current policy
            action, log_prob = self.actor.sample(state)

            # Get Q-values for sampled actions
            q1 = self.critic_1(state)
            q2 = self.critic_2(state)
            q = tf.minimum(q1, q2)

            # Actor loss: maximize Q-value while regularizing with entropy
            actor_loss = tf.reduce_mean(self.alpha * log_prob - q)

        # Optimize actor
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # Update target networks
        self.update_target_networks()

        return {
            'critic1_loss': float(critic1_loss),
            'critic2_loss': float(critic2_loss),
            'actor_loss': float(actor_loss)
        }

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update critic 1 target
        for target_var, source_var in zip(self.critic_1_target.variables, self.critic_1.variables):
            target_var.assign((1 - tau) * target_var + tau * source_var)

        # Update critic 2 target
        for target_var, source_var in zip(self.critic_2_target.variables, self.critic_2.variables):
            target_var.assign((1 - tau) * target_var + tau * source_var)

    def save_model(self, path):
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save actor model
        self.actor.save_weights(os.path.join(path, "actor.h5"))

        # Save critic models
        self.critic_1.save_weights(os.path.join(path, "critic_1.h5"))
        self.critic_2.save_weights(os.path.join(path, "critic_2.h5"))

        print(f"Model saved to {path}")

    def load_model(self, path):
        # Load actor model
        self.actor.load_weights(os.path.join(path, "actor.h5"))

        # Load critic models
        self.critic_1.load_weights(os.path.join(path, "critic_1.h5"))
        self.critic_2.load_weights(os.path.join(path, "critic_2.h5"))

        # Update target networks
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())

        print(f"Model loaded from {path}")


def train_agent(env, agent, num_episodes=30, batch_size=64, eval_freq=50, save_path="./models"):
    # Create result directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{save_path}/{timestamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Lists to store results
    rewards_history = []
    avg_rewards_history = []
    eval_rewards_history = []
    temperature_history = []
    humidity_history = []
    energy_history = []

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_temps = []
        episode_humidity = []
        episode_energy = []

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Store in replay buffer
            agent.replay_buffer.add(state, action, next_state, reward, done)

            # Train agent
            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            # Update state and collect metrics
            state = next_state
            episode_reward += reward
            episode_temps.append(env.temperature)
            episode_humidity.append(env.humidity)
            episode_energy.append(info['power_consumption'])

        # Record episode results
        rewards_history.append(episode_reward)
        temperature_history.append(np.mean(episode_temps))
        humidity_history.append(np.mean(episode_humidity))
        energy_history.append(np.mean(episode_energy))

        # Calculate moving average
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            print(f"Avg Temp: {np.mean(episode_temps):.2f}, Avg Humidity: {np.mean(episode_humidity):.2f}, Avg Energy: {np.mean(episode_energy):.2f}")

        # Evaluate agent
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(env, agent, num_eval_episodes=5)
            eval_rewards_history.append(eval_reward)
            print(f"Evaluation at episode {episode+1}: {eval_reward:.2f}")

            # Save model
            agent.save_model(f"{save_dir}/checkpoint_{episode+1}")

            # Plot training progress
            plot_training_progress(rewards_history, avg_rewards_history, eval_rewards_history,
                                  temperature_history, humidity_history, energy_history,
                                  save_dir=save_dir)

    # Save final model
    agent.save_model(f"{save_dir}/final_model")

    return rewards_history, avg_rewards_history, eval_rewards_history


def evaluate_agent(env, agent, num_eval_episodes=10):
    eval_rewards = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select action (deterministic policy for evaluation)
            action = agent.select_action(state, evaluate=True)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Update state and reward
            state = next_state
            episode_reward += reward

        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


def plot_training_progress(rewards, avg_rewards, eval_rewards, temperatures, humidities, energies, save_dir=None):
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot rewards
    axs[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    axs[0, 0].plot(avg_rewards, label='Avg Reward (100 ep)')

    if eval_rewards:
        # Mark evaluation points on x-axis
        eval_x = np.linspace(0, len(rewards)-1, len(eval_rewards), dtype=int)
        axs[0, 0].plot(eval_x, eval_rewards, 'ro', label='Evaluation')

    axs[0, 0].set_title('Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot temperatures
    axs[0, 1].plot(temperatures)
    axs[0, 1].axhline(y=20, color='g', linestyle='--', label='Min Optimal')
    axs[0, 1].axhline(y=25, color='g', linestyle='--', label='Max Optimal')
    axs[0, 1].axhline(y=30, color='r', linestyle='--', label='Critical')
    axs[0, 1].set_title('Average Temperature per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Temperature (°C)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot humidities
    axs[1, 0].plot(humidities)
    axs[1, 0].axhline(y=40, color='g', linestyle='--', label='Min Optimal')
    axs[1, 0].axhline(y=60, color='g', linestyle='--', label='Max Optimal')
    axs[1, 0].axhline(y=70, color='r', linestyle='--', label='Critical')
    axs[1, 0].set_title('Average Humidity per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Humidity (%)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot energy consumption
    axs[1, 1].plot(energies)
    axs[1, 1].set_title('Average Energy Consumption per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Energy Consumption')
    axs[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure if directory is provided
    if save_dir:
        plt.savefig(f"{save_dir}/training_progress.png")

    # Display figure
    plt.show()


# Main training function
def main():
    # Set up environment and agent
    env = ServerRoomEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.2  # Entropy regularization coefficient
    )

    # Train agent
    rewards, avg_rewards, eval_rewards = train_agent(
        env=env,
        agent=agent,
        num_episodes=40,
        batch_size=64,
        eval_freq=50,
        save_path="./server_room_models"
    )

    # Final evaluation
    final_eval_reward = evaluate_agent(env, agent, num_eval_episodes=10)
    print(f"Final evaluation reward: {final_eval_reward:.2f}")

    # Create server room visualization showing temperature/humidity control over time
    visualize_control(env, agent)


def visualize_control(env, agent, num_steps=20):
    """Visualize the agent's control of the server room over time."""
    # Reset environment
    state = env.reset()

    # Initialize arrays to store data
    temperatures = []
    humidities = []
    fan_states = []
    fan_speeds = []
    server_loads = []
    rewards = []

    # Run simulation
    for step in range(num_steps):
        # Select action
        action = agent.select_action(state, evaluate=True)

        # Take action
        next_state, reward, done, info = env.step(action)

        # Store data
        temperatures.append(env.temperature)
        humidities.append(env.humidity)
        fan_states.append(action[0] > 0.5)
        fan_speeds.append(action[1])
        server_loads.append(env.server_load)
        rewards.append(reward)

        # Update state
        state = next_state

        # Check if episode is done
        if done:
            break

    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Plot temperature
    axs[0].plot(temperatures, label='Temperature')
    axs[0].axhline(y=env.optimal_temp_range[0], color='g', linestyle='--', label='Min Optimal')
    axs[0].axhline(y=env.optimal_temp_range[1], color='g', linestyle='--', label='Max Optimal')
    axs[0].axhline(y=env.critical_temp, color='r', linestyle='--', label='Critical')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Server Room Control Visualization')
    axs[0].legend()
    axs[0].grid(True)

    # Plot humidity
    axs[1].plot(humidities, label='Humidity')
    axs[1].axhline(y=env.optimal_humidity_range[0], color='g', linestyle='--', label='Min Optimal')
    axs[1].axhline(y=env.optimal_humidity_range[1], color='g', linestyle='--', label='Max Optimal')
    axs[1].axhline(y=env.critical_humidity, color='r', linestyle='--', label='Critical')
    axs[1].set_ylabel('Humidity (%)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot fan action
    axs[2].step(range(len(fan_states)), fan_states, label='Fan On/Off', where='post')
    axs[2].plot(fan_speeds, label='Fan Speed', alpha=0.7)
    axs[2].set_ylabel('Fan Control')
    axs[2].set_ylim(-0.1, 1.1)
    axs[2].legend()
    axs[2].grid(True)

    # Plot server load and rewards
    ax3 = axs[3]
    ax3.plot(server_loads, label='Server Load', color='purple')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Server Load')
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # Add rewards on secondary y-axis
    ax4 = ax3.twinx()
    ax4.plot(rewards, label='Reward', color='orange', alpha=0.6)
    ax4.set_ylabel('Reward')
    ax4.legend(loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.savefig("server_room_control.png")
    plt.show()


if __name__ == "__main__":
    main()


# After training, let's test the agent

def test_agent():
    # Reset environment for testing
    state = env.reset()
    total_reward = 0

    for step in range(100):  # Run for 100 steps
        # Use the trained agent to select an action
        action = agent.select_action(state)

        # Fan on/off and fan speed are in the action
        fan_on = action[0] > 0.5  # Fan on if action[0] > 0.5
        fan_speed = action[1]  # Speed is between 0 and 1

        # Print the fan status and speed
        print(f"Step {step+1} - Fan On: {fan_on}, Fan Speed: {fan_speed:.2f}")

        # Take the action in the environment
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Total Test Reward: {total_reward}")

# Run the test after training
test_agent()
