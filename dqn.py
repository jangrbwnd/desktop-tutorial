import cv2
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from game_ai import Breakout
import os
import sys
import time
import pygame

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.01
        self.initial_epsilon = 0.2
        self.number_of_iterations = 4000000
        self.replay_memory_size = 750000
        self.minibatch_size = 32
        self.explore = 3000000 # Timesteps to go from INITIAL_EPSILON to FINAL_EPSILON

        self.conv1 = nn.Conv2d(4, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #make sure input tensor is flattened
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


def preprocessing(image):
	image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
	image_data[image_data > 0] = 255
	image_data = np.reshape(image_data,(84, 84, 1))
	image_tensor = image_data.transpose(2, 0, 1)
	image_tensor = image_tensor.astype(np.float32)
	image_tensor = torch.from_numpy(image_tensor)
	return image_tensor

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # initialize mean squared error loss
    criterion = nn.MSELoss() # crossentropy

    # instantiate game
    game_state = Breakout()

    # initialize replay memory
    D = deque()
    #replay = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 0
    image_data, reward, terminal = game_state.take_action(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0) # 1-4-84-84

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    #epsilon = 0.0927
    #iteration = 420000
    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0] # Output size = torch.Size([2]) tensor([-0.0278,  1.7244]
        #output = model(state)

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Random action!")

        # Pick action --> random or index of maximum q value
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        #print("Action index shape: ", action_index.shape) # torch.Size([])
       
        action[action_index] = 1

        if epsilon > model.final_epsilon:
            epsilon -= (model.initial_epsilon - model.final_epsilon) / model.explore

        # get next state and reward
        image_data_1, reward, terminal = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        
        #print("İmage data_1 shape: ", image_data_1.shape)  # 1-84-84

        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)   # squeeze(0).shape = 4-84-84
        #print("State_1 Shape: ", state_1.shape) # State_1 Shape = ([1, 4, 84, 84])     # squeeze(0)[1:,:,:].shape = 3-84-84
        action = action.unsqueeze(0)
        #print("Action size: ", action.shape) # 1-2
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)   
        #print("Reward size: ", reward.shape)
        # save transition to replay memory
        D.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(D) > model.replay_memory_size:
            D.popleft()

        # sample random minibatch
        # it picks k unique random elements, a sample, from a sequence: random.sample(population, k)
        minibatch = random.sample(D, min(len(D), model.minibatch_size))
        # unpack minibatch

        state_batch   = torch.cat(tuple(d[0] for d in minibatch))
        #print("state_batch size: ", state_batch.shape)
        action_batch  = torch.cat(tuple(d[1] for d in minibatch))
        #print("action_batch size: ", action_batch.shape)
        reward_batch  = torch.cat(tuple(d[2] for d in minibatch))
        #print("reward_batch size: ", reward_batch.shape)
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
        #print("state_1_batch size: ", state_1_batch.shape)
        
        # get output for the next state
        output_1_batch = model(state_1_batch)
        #print("output_1_batch: " , output_1_batch.shape)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q) Target Q value Bellman equation.
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        
        # extract Q-value -----> column1 * column1 + column2 * column2
        # The main idea behind Q-learning is that if we had a function Q∗ :State × Action → ℝ
        #that could tell us what our return would be, if we were to take an action in a given state,
        #then we could easily construct a policy that maximizes our rewards
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        #print("q_value: ", q_value.shape)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 10000 == 0:
            torch.save(model, "current_model_" + str(iteration) + ".pth")
              
        print("total iteration: {} Elapsed time: {:.2f} epsilon: {:.5f}"
               " action: {} Reward: {:.1f}".format(iteration,((time.time() - start)/60),epsilon,action_index.cpu().detach().numpy(),reward.numpy()[0][0]))

def test(model):
    pygame.init()
    game_state = Breakout()
    SIZE_OF_THE_SCREEN = 600, 645
    screen = pygame.display.set_mode(SIZE_OF_THE_SCREEN)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    screen.fill(BLACK)
    font = pygame.font.SysFont(None, 30)
    for i in range(3, 0, -1):
        screen.fill(BLACK)
        countdown_text = font.render(f'{i}', True, WHITE)
        screen.blit(countdown_text, (SIZE_OF_THE_SCREEN[0] // 2, SIZE_OF_THE_SCREEN[1] // 2))
        pygame.display.update()
        time.sleep(1)

    # initial action is do nothing
    running = True
    score_list=[]
    while running:
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[0] = 1
        image_data, reward, terminal = game_state.take_action(action)
        score = game_state.score
        if len(score_list) < 2:
            score_list.append(score)
        elif len(score_list) ==2:
            score_list[0] = score_list[1]
            score_list[1] = score
        image_data = preprocessing(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
        
        # get output from the neural network
        output = model(state)[0]

         # get action
        action_index = torch.argmax(output)
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1
        if terminal:
            # "Finish" 메시지 표시
            game_over_text = font.render('Finish', True, WHITE)
            screen.blit(game_over_text, (SIZE_OF_THE_SCREEN[0] // 2 - 60, SIZE_OF_THE_SCREEN[1] // 2))
            pygame.display.update()
            
            # 사용자가 창을 닫을 때까지 대기
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        quit()
                score_text = font.render(f'Score: {score_list[0]}', True, WHITE) 
                screen.blit(score_text, (SIZE_OF_THE_SCREEN[0] // 2 - 60, SIZE_OF_THE_SCREEN[1] // 2 + 30))
                
                pygame.display.update()




def main(mode):
    if mode == 'test':
        model = torch.load('current_model_20000.pth', map_location='cpu').eval()
        test(model)

    
    elif mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')
        model = NeuralNetwork()
        model.apply(init_weights)
        start = time.time()
        train(model, start)
    elif mode == 'continue': # You can change trained model id and keep training.
        model = torch.load('current_model_90000.pth', map_location='cpu').eval()#90000쓰거나 1,2,3,4만 중 괜찮은거 찾기

        start = time.time()
        train(model, start)




if __name__ == "__main__":
    #main(sys.argv[1])
    main('test')