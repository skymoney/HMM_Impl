#-*- coding:utf-8 -*-

'''
A simple HMM(Hidden Markov Model) Demo

Author: Cheng@NJU
Date: 2014/06/26
'''

##A simple demo 
# predict weather by using observation man states

'''
test set

M= 4  #observation states
N= 3　#hidden states
A:	#transmition matrix
0.500 0.375 0.125
0.250 0.125 0.625
0.250 0.375 0.375
B:	#mixture matrix
0.60 0.20 0.15 0.05
0.25 0.25 0.25 0.25
0.05 0.10 0.35 0.50
pi:
0.63 0.17 0.20
'''

#hidden state of HMM
hidden_states = {'Sunny', 'Cloudy', 'Rainy'}

#observation states
obv_states = {''}

#PI vector
pi = []

#transmition matrix
A = []

#confusion matrix
B = []

#basic definition
class HMM(object):
	def __init__(self, pi=[], A=[], B=[], hidden_states=[], obv_states=[]):
		self.pi = pi
		self.A = A
		self.B = B
		self.hidden_states = hidden_states
		self.obv_states = obv_states

	def set_pi(self, new_pi):
		self.pi = new_pi

	def set_transmition_matrix(self, new_t_m):
		self.A = new_t_m

	def set_confusion_matrix(self, new_c_m):
		self.B = new_c_m

	def set_hidden_states(self, new_h_s):
		self.hidden_states = new_h_s

	def set_obv_states(self, new_o_s):
		self.obv_states = new_o_s

#use forward algo to calculate first queetion in HMM

# given s seq O = [O1,...OT] and model, calculate the 
# pro of the seq

def forward(hmm_model, seq=[]):
	#hmm_model: hmm model
	#seq: observation sequence
	'''
	use Forward Algo to calculate the probability of given seq
	'''
	alpha = []
	alpha.append([]) #reserve empty
	#initialize
	alpha.append([])
	for i in range(len(hmm_model.hidden_states)):
		alpha[1].append(hmm_model.pi[i]*hmm_model.B[i][hmm_model.obv_states.index(seq[0])])

	#induction
	for t in range(1, len(seq)):
		alpha.append([])

		for i in range(len(hmm_model.hidden_states)):
			#at time t the state i
			current_sum = 0.0
			for j in range(len(hmm_model.hidden_states)):
				#at time t the target state j
				current_sum += alpha[t][j]*hmm_model.A[j][i]

			alpha[t+1].append(current_sum*hmm_model.B[i][hmm_model.obv_states.index(seq[t])])


	#termination
	#print sum(alpha[len(seq)])
	return sum(alpha[len(seq)])

# given a seq O = [O1,...OT] and model, calculate the 
#most possiable states

def viterbi(hmm, seq=[]):
	'''
	calculate the most possiable hidden seq given observation seq
	by using Viterbi Algo
	'''
	delta = [] #delta value for recoding local probability 
	psi = []

	#initialize
	delta.append([]) #leave zero empty
	delta.append([]) #add first array
	psi.append([])
	psi.append([])
	for i in range(len(hmm.hidden_states)):
		delta[1].append(hmm.pi[i] * hmm.B[i][hmm.obv_states.index(seq[0])])
		psi[1].append(0)

	#Recursion
	for t in range(2, len(seq)+1):
		delta.append([])
		psi.append([])
		
		for i in range(len(hmm.hidden_states)):
			delta[t].append([])
			psi[t].append([])			
			max_val, max_val_idx= 0.0, 1
			for j in range(len(hmm.hidden_states)):
				val = delta[t-1][j] * hmm.A[j][i]

				if val > max_val:
					max_val = val
					max_val_idx = j
			
			delta[t][i] = max_val * hmm.B[i][hmm.obv_states.index(seq[t-1])]
			psi[t][i] = max_val_idx

	#Termination
	prop = 0.0
	path_idx = [] #record path index
	path_idx.append(1)
	for i in range(len(hmm.hidden_states)):
		if delta[len(seq)][i] > prop:
			prop = delta[len(seq)][i]
			path_idx[0] = i

	#path track
	for i in range(len(seq)-1):
		path_idx.append(psi[len(seq)-i][path_idx[i]])

	return map(lambda x: hmm.hidden_states[x], path_idx[::-1])