#Value Iteration on Grid World MDP

import traceback

#Define the class for generic gridworld
class GridWorld:
	
	def __init__(self, num_rows, num_columns, walls, terminal_states, reward, transition, discount_factor, epsilon ):
		self.num_rows = num_rows
		self.num_columns = num_columns
		self.walls = walls
		self.terminal_states = terminal_states
		self.reward = reward
		self.transition = transition
		self.discount_factor = discount_factor
		self.epsilon = epsilon
	
	#Function to generate next state based on some action, if wall is encountered , agent stays in its current cell
	def move(self,state,action):
		if(action=='W'):
			if(state.x>1):
				if State(state.x-1,state.y) not in self.walls:
					return State(state.x-1,state.y)
		if(action=='E'):
			if(state.x < self.num_columns):
				if State(state.x+1,state.y) not in self.walls:
					return State(state.x+1,state.y)
		if(action=='S'):
			if(state.y>1):
				if State(state.x,state.y-1) not in self.walls:
					return State(state.x,state.y-1)
		if(action=='N'):
			if(state.y < self.num_rows):
				if State(state.x,state.y+1) not in self.walls:
					return State(state.x,state.y+1)
		return state
				
			
class State:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def __eq__(self, other): 
		return self.x == other.x and self.y == other.y
	
	def __repr__(self):
		return "x= "+str(self.x)+" y= "+str(self.y)
		
	def __hash__(self):
		return hash((self.x, self.y))

		

class MDP:
	
	def __init__(self, states, actions, transitions, rewards,discount_factor,epsilon,gw):
		self.states = states
		self.actions = actions
		self.transitions = transitions 
		self.rewards = rewards
		self.discount_factor = discount_factor
		self.epsilon = epsilon
		self.gw = gw
		
		
	def value_iteration(self):
		it = 0
		utility_dict = {}
		policy_dict = {}
		for state in self.states:
			if state in self.gw.walls:
				utility_dict[state] = '--------------'
			else:
				utility_dict[state] = 0
			
		while True:
			print("iteration: " + str(it))
			print_in_grid(utility_dict,self.gw.num_rows,self.gw.num_columns)
			utility_dict_previous = utility_dict.copy()
			delta = 0
			for state in self.states:
				print(state)
				if state in self.gw.walls :  
					continue
				immediate_reward = self.rewards[state]
				#future_rewards = {}
				max_future_reward = -99999
				policy_dict[state] = 'T' #default terminal
				if state in self.gw.terminal_states : 
					max_future_reward=0
				else:
					for action in self.actions:
						future_reward_for_current_action = 0
						for transition in self.transitions[state][action]:
							prob = transition[0]
							utility_next_state = utility_dict_previous[transition[1]]
							future_reward_for_current_action+= prob * utility_next_state
						if future_reward_for_current_action > max_future_reward:
							max_future_reward = future_reward_for_current_action
							policy_dict[state] = action
				utility_dict[state] = immediate_reward + self.discount_factor * max_future_reward	
				if abs(utility_dict[state] - utility_dict_previous[state]) > delta:
					delta = abs(utility_dict[state] - utility_dict_previous[state])
			if delta <= self.epsilon*(1-self.discount_factor)/self.discount_factor:
				print("Final Value After Convergence")
				print_in_grid(utility_dict,self.gw.num_rows,self.gw.num_columns)
				print("Final Policy")
				print_in_grid(policy_dict,self.gw.num_rows,self.gw.num_columns)
				break
						
			#if it>1: break
			it+=1
			

def print_in_grid(val_dict,num_rows,num_columns):
	for i in range(num_rows,0,-1):
		for j in range(1,num_columns+1):
			if State(j,i) not in val_dict: print('-'+" "),
			else: print(str(val_dict[State(j,i)]) + " "),
		print("\n")
	pass
	
			
def parse_input(input_file):
	grid_world_dict = {}
	with open(input_file,"r") as fopen:
		for line in fopen:
			if ':' in line:
				key,value = line.strip().split(':')
				grid_world_dict[key.strip()] = value.strip()
	try:
		num_rows = grid_world_dict['size'].split(" ")[1]
		num_columns = grid_world_dict['size'].split(" ")[0]
		walls = grid_world_dict['walls'].split(",")
		terminal_states = grid_world_dict['terminal_states'].split(",")
		non_terminal_rewards = grid_world_dict['reward']
		transition = grid_world_dict['transition_probabilities']
		discount_factor = grid_world_dict['discount_rate']
		epsilon = grid_world_dict['epsilon']
		wall_list =[]
		for wall in walls:
			wall_list.append(State(int(wall.strip().split(" ")[0]),int(wall.strip().split(" ")[1])))
		terminal_state_dict = {}	
		for terminal_state in terminal_states:
			
			x,y,reward = terminal_state.strip().split(" ")
			terminal_state_dict[State(int(x),int(y))] = float(reward)
		print(int(num_rows), int(num_columns), wall_list, terminal_state_dict, float(non_terminal_rewards), transition, float(discount_factor), float(epsilon))
		return GridWorld(int(num_rows), int(num_columns), wall_list, terminal_state_dict, float(non_terminal_rewards), transition, float(discount_factor), float(epsilon))
		
	except Exception as e:
		traceback.print_exc()
		print("invalid input file")
	    #return GridWorld()		

def create_MDP(grid_world):
	actions = ['N','E','S','W']
	states = []
	for i in range(0,grid_world.num_rows):
		for j in range(0,grid_world.num_columns):
			states.append(State(j+1,i+1)) #grid with (1,1) at bottom left
	reward_dict = {}
	
	#define reward function
	for state in states:
		if state in grid_world.terminal_states:
			reward_dict[state] = grid_world.terminal_states[state]
		else:
			reward_dict[state] = grid_world.reward

			
	#define transition function
	transition_dict = {}
	for state in states:
		
		transition_dict[state] = {}
		for action in actions:
			orthogonal_action_left = actions[(actions.index(action)-1) % len(actions)]
			orthogonal_action_right = actions[(actions.index(action)+1) % len(actions)]
			transition_dict[state][action] = [(0.8, grid_world.move(state,action)), (0.1, grid_world.move(state,orthogonal_action_left)),(0.1, grid_world.move(state,orthogonal_action_right))]
			#cannot transition from terminal_states
			if state in grid_world.terminal_states:
				transition_dict[state][action] = [(0.0, grid_world.move(state,action)), (0.0, grid_world.move(state,orthogonal_action_left)),(0.0, grid_world.move(state,orthogonal_action_right))]
	return MDP(states,actions,transition_dict,reward_dict,grid_world.discount_factor,grid_world.epsilon,grid_world)
	

	
	
def main():
	grid_world = parse_input("mdp_input.txt")
	grid_mdp = create_MDP(grid_world)
	grid_mdp.value_iteration()
	
if __name__=="__main__": main()
