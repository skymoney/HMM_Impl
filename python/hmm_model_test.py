#-*- coding:utf-8 -*-

import hmm_model_demo

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

if __name__ == '__main__':
	hmm = hmm_model_demo.HMM()

	print 'Forward Algo'
	hmm.set_pi(new_pi = [0.63, 0.17, 0.20])
	hmm.set_transmition_matrix(new_t_m=[[0.5, 0.375, 0.125], 
		[0.25, 0.125, 0.625], 
		[0.25, 0.375, 0.375]])
	hmm.set_confusion_matrix(new_c_m=[[0.6, 0.2, 0.15, 0.05], 
		[0.25, 0.25, 0.25, 0.25], 
		[0.05, 0.10, 0.35, 0.50]])
	hmm.set_hidden_states(new_h_s=['Sunny', 'Cloudy', 'Rainy'])
	hmm.set_obv_states(new_o_s=['Dry', 'Dryish', 'Damp', 'Soggy'])

	print hmm_model_demo.forward(hmm, seq=['Dry', 'Damp', 'Soggy'])

	print 'viterbi Algo'
	
	hmm.set_pi(new_pi = [0.333, 0.333, 0.333])
	hmm.set_transmition_matrix(new_t_m = [[0.333, 0.333, 0.333], 
		[0.333, 0.333, 0.333], 
		[0.333, 0.333, 0.333]])
	hmm.set_confusion_matrix(new_c_m = [[0.5, 0.5], 
		[0.75, 0.25], 
		[0.25, 0.75]])
	hmm.set_obv_states(new_o_s = ['OK', 'Bad'])
	hmm.set_hidden_states(new_h_s = ['H1', 'H2', 'H3'])
	
	print hmm_model_demo.viterbi(hmm, seq=['OK', 'OK', 'OK', 'OK', 'Bad', 'OK', 'Bad', 'Bad', 'Bad', 'Bad'])
	