import numpy as np
import matplotlib.pyplot as plt
from DroneDeliveryEnvironment import DroneDeliveryEnvironment

class DroneDeliveryTrainer:
    def __init__(self, env, num_episodes=4000, alpha=0.1, gamma=0.9, epsilon=0.5):
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env.epsilon = epsilon  # sincronizza l'epsilon dell'ambiente

    def train(self):
        rewards_per_episode = []

        for episode in range(self.num_episodes):
            state = self.env.reset()[0]  # reset del drone al suo stato iniziale
            done = False  # stato di completamento del drone
            total_reward = 0

            while not done:
                action = self.env.choose_action(state)  # sceglie l'azione
                next_state, reward, done = self.env.step(0, action)  # esegue uno step

                # Aggiorna la q-table
                self.env.update_q_table(state, action, reward, next_state)

                state = next_state
                total_reward += reward

                # batteria scarica
                if next_state[2] == 0:
                    done = True
                    #print(
                    #    f"Battery depleted for Drone in episode {episode}. State: {next_state}")

            rewards_per_episode.append(total_reward)

            # riduce il valore di epsilon gradualmente, per favorire l'addestramento
            self.epsilon = max(0.01, self.epsilon * 0.997)
            # aggiorna epsilon nell'ambiente
            self.env.epsilon = self.epsilon

            # print(f"Episode {episode}/{self.num_episodes} complete, total_reward Reward: {total_reward}")
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"Episode {episode}/{self.num_episodes} complete, Average Reward: {avg_reward}")

        # salva la q-table al termine dell'addestramento
        np.save("q_table.npy", self.env.Q_table)
        print("Q-table salvata come 'q_table.npy'.")

        # rappresenta i risultati dell'addestramento graficamente
        avg_rewards = [np.mean(rewards_per_episode[i:i + 100]) for i in range(0, len(rewards_per_episode), 100)]
        plt.plot(range(0, len(rewards_per_episode), 100), avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Cumulative Reward')
        plt.title('Average Cumulative Reward per 100 Episodes')
        plt.show()

def main():
    grid_size = (5, 5)
    env = DroneDeliveryEnvironment(grid_size, 1,
                                   training_mode=True)

    print("Starting training...")
    trainer = DroneDeliveryTrainer(env)
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
