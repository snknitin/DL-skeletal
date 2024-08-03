# Example usage
import gym
import yaml

if __name__ == "__main__":
    # Load your configuration
    with open('path_to_your_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Add any additional parameters
    config['reward_func'] = your_reward_function  # Define this separately
    config['fc_lt_mean'] = [5, 5, 5]  # Example values
    config['fc_lt_var'] = [1, 1, 1]  # Example values

    # Create the environment
    # Now you can create your environment
    env = gym.make('MultiSKU-v0', env_cfg=config)

    # Use the environment
    observation = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # Your agent here (e.g., model.predict(observation))
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()