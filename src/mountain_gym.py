import json, time;
import numpy as np;
import keras, gym, agent;

def sign(value):
	return round(value/abs(value)) if value != 0 else 0;

def mean(values):
	return round(sum(values) / len(values), 2) if type(values) == list and len(values) > 0 else 0.0;

if __name__ == "__main__":
	env = gym.make("MountainCar-v0");
	state_size = env.observation_space.shape[0];
	
	model_name = input("Model name -> ");
	load_trained = input("Load trained (y/n)? ");
	load_trained = load_trained.lower() == "y";
	
	my_model_location = "models/" + model_name + "/";
	my_model = my_model_location + ("model_trained.h5" if load_trained else "model.h5");

	epsilon = input("Epsilon -> ");
	
	print("Loading", my_model, "with epsilon", epsilon);
	agent = agent.DQNAgent(my_model);
	
	try: agent.memory = json.load(my_model_trained.replace(".h5", ".json"));
	except: agent.memory = [];

	episode_count = int(input("Episode count -> "));
	batch_size = 16;
	
	max_score = None;
	highest_score = 0;
	scores = [];
	rewards = [];
	
	start = time.time();
	first_start = start;
	
	for e in range(episode_count):		
		# at each episode, reset environment to starting position
		state = env.reset();
		state = np.reshape(state, [1, state_size]);
		score = 0;
		
		done = False;
		rewards.append(0.0);
		
		while not done and (score < max_score if max_score else True):
			# show game graphics
			# env.render();
			
			# select action, observe environment, calculate reward
			action = agent.act(state);
			next_state, reward, done, _ = env.step(action);
			next_state = np.reshape(next_state, [1, state_size]);
			score += 1;
			
			# reward -= 0.5 - next_state[0][0];
			# reward += 1-np.exp(-0.5 - next_state[0][0]);
			# reward += (((next_state[0][0] + 1.2)/1.8)**4)*2.0;
			# print(reward)
			
			# save experience and update current state
			agent.remember(state, action, reward, next_state, done);
			state = next_state;
			rewards[-1] += reward;
			
			# dynamic batch_size and max_memory
			# batch_size = round((highest_score/500) * 80) + 48;
			# max_memory = round(highest_score*20 + 250) if highest_score != 500 else 9500;
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size);
		
		scores.append(score);
		if len(scores) > 20: scores = scores[-20:];
		if len(rewards) > 20: rewards = rewards[-20:];
		
		print("episode: {}/{}, reward: {}, last 20 average: {}, e: {:.2}, in memory: {}"
				.format(e+1, episode_count, round(rewards[-1], 2), round(mean(rewards), 2), agent.epsilon, len(agent.memory)));
		
		if score < 200:
			agent.save();
		
		# if len(scores) >= 5 and score == 500 and sum(scores[-5:]) >= 1500:
			# agent.save();
		
		# if score < 200 and len(scores) >= 15 and sum(i < 200 for i in scores[-15:]) == 15:
			# print("training successfull!");
			# agent.save("final");
			# break;
			
		if rewards[-1] >= -100 and len(rewards) >= 15 and sum(i >= -100 for i in rewards[-15:]) == 15:
			print("training successfull!");
			agent.save("final");
			break;
			
		# if score > highest_score: highest_score = score;
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();
			agent.merge_models();

	agent.save();
	print("Total training time:", round((time.time()-first_start)/60, 2), "minutes");