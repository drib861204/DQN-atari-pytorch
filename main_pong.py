import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import gym
from env_wrapper import PongEnvWrapper
import dqn
import torch
import models
import argparse
import time
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from Pendulum_v2 import *  # added by Ben

RENDER = 0
SEED = 0
SAVED_MODEL = None
TRIAL = 21

def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.05, epsilon_decay=100000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

def save_gif(img_buffer, fname, gif_path="gif"):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    img_buffer[0].save(os.path.join(gif_path, fname), save_all=True, append_images=img_buffer[1:], duration=1, loop=1)

def train(env, agent, save_path="save", max_steps=1000000):
    total_step = 0
    episode = 0
    scores_window = deque(maxlen=100)
    while True:
        # Reset environment.
        state = env.reset(SAVED_MODEL, SEED)

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0.

        # One episode.
        while True:
            # Select action.
            epsilon = epsilon_compute(total_step)
            action = agent.choose_action(state, epsilon)

            # Get next stacked state.
            state_next, reward, done, info = env.step(action)

            # Store transition and learn.
            total_reward += reward
            agent.store_transition(state, action, reward, state_next, done)
            if total_step > 4*agent.batch_size:
                loss = agent.learn()

            state = state_next.copy()
            step += 1
            total_step += 1

            if total_step % 100 == 0 or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f}'\
                    .format(episode, step, total_step, reward, total_reward, loss, epsilon), end="")
            
            if total_step % 10000 == 0:
                print("\nSave Model ...")
                agent.save_load_model(op="save", path=save_path, fname=f"qnet_{TRIAL}.pt")
                play(env, agent)
                #print("Generate GIF ...")
                #img_buffer = play(env, agent, stack_frames, img_size)
                #save_gif(img_buffer, "train_" + str(total_step).zfill(6) + ".gif")
                #print("Done !!")

            if done or step>2000:
                scores_window.append(total_reward)
                writer.add_scalar("Average100", np.mean(scores_window), total_step)
                episode += 1
                print()
                break
        
        if total_step > max_steps:
            break

def play(env, agent):
    # Reset environment.
    state = env.reset(SAVED_MODEL, SEED)
    #img_buffer = [Image.fromarray(state[0]*255)]

    # Initialize information.
    step = 0
    total_reward = 0
    #loss = 0.
    state_action_log = np.zeros((1, 4))

    # One episode.
    while True:
        # Select action.
        action = agent.choose_action(state, 0)

        # Get next stacked state.
        state_next, reward, done, info = env.step(action)
        #if step % 2 == 0:
        #    img_buffer.append(Image.fromarray(state_next[0]*255))
        if RENDER:
            env.render(1) # Cant't use in colab.

        state_action = np.append(state, action)
        state_action_log = np.concatenate((state_action_log, np.asmatrix(state_action)), axis=0)

        # Store transition and learn.
        total_reward += reward
        print('\rStep: {:3d} | Reward: {:.3f} / {:.3f}'\
            .format(step, reward, total_reward), end="")
            
        state = state_next.copy()
        step += 1
        if done or step>5000:
            print()
            break

    #return img_buffer
    if RENDER:
        fig, axs = plt.subplots(4)
        fig.suptitle('DDQN Transient Response')
        t = np.arange(0, env.dt * np.shape(state_action_log)[0], env.dt)
        axs[0].plot(t[1:], state_action_log[1:, 0])
        axs[3].plot(t[1:], state_action_log[1:, 1])
        axs[1].plot(t[1:], state_action_log[1:, 2])
        axs[2].plot(t[1:], state_action_log[1:, 3] * env.max_torque)
        axs[0].set_ylabel('q1(rad)')
        axs[1].set_ylabel('q2 dot(rad/s)')
        axs[2].set_ylabel('torque(Nm)')
        axs[3].set_ylabel('q1 dot(rad/s)')
        axs[2].set_xlabel('time(s)')
        # axs[0].set_ylim([-0.01,0.06])
        # axs[0].set_ylim([-pi-0.5,pi+0.5])
        #axs[1].set_ylim([-34, 34])
        #axs[2].set_ylim([-12, 12])
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', nargs='?', type=str, default="train", help='train / test')

    #train_test = parser.parse_args().type
    args = parser.parse_args()
    train_test = args.type

    if train_test == "train":
        writer = SummaryWriter(f"runs/rwip_{TRIAL}")

    #stack_frames = 4
    #img_size = (84,84)
    #env_name = "PongNoFrameskip-v4"
    #env_ = gym.make(env_name)
    #env = PongEnvWrapper(env_, k=stack_frames, img_size=img_size)
    env = Pendulum(RENDER, SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    agent = dqn.DeepQNetwork(
        n_actions = env.action_space.n,
        #input_shape = [stack_frames, *img_size],
        input_shape = env.observation_space.shape[0],
        qnet = models.QNet,
        device = device,
        learning_rate = 2e-4,
        reward_decay = 0.99,
        replace_target_iter = 1000,
        memory_size = 10000,
        batch_size = 32,
        )

    t0 = time.time()

    if train_test == "train":
        train(env, agent, f"runs/rwip_{TRIAL}", max_steps=250000)
        t1 = time.time()
        timer(t0, t1)
        writer.close()
    elif train_test == "test":
        agent.save_load_model(op="load", path=f"runs/rwip_{TRIAL}", fname=f"qnet_{TRIAL}.pt")
        for _ in range(3):
            play(env, agent)
        #img_buffer = play(env, agent, stack_frames, img_size)
        #save_gif(img_buffer, "test.gif")
    else:
        print("Wrong args.")