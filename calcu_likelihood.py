# coding=utf-8

###############################################################################
#Description:
# calculate Log-likelihood of DNN-HMM model
# require n_score from Julius 'state alignmemt' result
# warning: this is not complete compatible with julius.
#
# Input:
#    time series of triphone and hidden state
#    time series of dnn output
# 
#    hmm_model
#    transition_matrix
#    dictionary_hmm_transition-matrix_hidden-state
#
# Output:
#     Log-likelihood normalized by number of frames
#
# This is based on C program in julius-4.4.2.zip. of
#
# Copyright (c) 1991-2016 Kawahara Lab., Kyoto University
# Copyright (c) 1997-2000 Information-technology Promotion Agency, Japan
# Copyright (c) 2000-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2016 Julius project team, Nagoya Institute of Technology
#
# License: see LICENSE-Julius.txt
#
#
# This use julius dictation-kit-v4.4 dnn model data.
# Please see LICENSE-Julius Dictation Kit.txt
#
#Date 2018-03-30
#By Shun
###############################################################################

import os
import sys
import numpy as np
import json

# Check version
# Windows 10 (64bit)
# Python 2.7.12 on win32 (Windows version)
# numpy (1.14.0)


class Class_calcu(object):
	def __init__(self,):
		self.hmm=self.npload('hmm_model_data0.npy')
		self.trans=self.npload('transition_matrix_data0.npy')
		self.dic=self.jload('dictionary_hmm_trans_hiddenstatenum0.json')
		
	
	def calcu(self, alignment, dnnout):
		#
		print('alignment length ', len(alignment))
		
		# check dnn output matrix shape
		if dnnout.shape[0] != 4874 :
			print('-warning: dnnout.shape[0] != 4874, matrix was transposed ', dnnout.shape[0])
			dnn0=dnnout.T
		else:
			dnn0=dnnout
		
		#ready 
		n_score=np.zeros(len(alignment) )
		am_score=0.0
		
		frame_start=np.zeros(len(alignment), dtype=np.int)
		frame_end=np.zeros(len(alignment), dtype=np.int)
		triphones=[]
		hidden_state=np.zeros(len(alignment), dtype=np.int)
		
		for i in range (len(alignment)):
			frame_start[i]=alignment[i][0][0]
			frame_end[i]=alignment[i][0][1]
			triphones.append(alignment[i][1])
			hidden_state[i]=alignment[i][2]
		
		# get dic_value as triphone is key
		# dic_value[ HMM model index, transition matrix index, hidden state number]
		dic_values= self.find_key_and_values(triphones)
		
		
		frames=0  # frame number counter
		for j in range (len(alignment)):
			
			for l in range (frame_start[j], (frame_end[j]+1) ):
				
				# transition probability
				if hidden_state[j] == 1 and l == frame_start[j] :  # find start of one triphone, It must start hidden state No.1 ?
					if frames == 0:  # no previous data, then set 0.0
						t0=0.0
					else:		# From Intermediate to Terminal of previous transition matrix
						tr_id=dic_values[j-1][1]
						tr0=self.trans[tr_id]
						hidden_state_num=dic_values[j-1][2]
						t0=tr0[ hidden_state_num  * hidden_state[j-1] + (hidden_state_num-1) ]
				else:
					tr_id=dic_values[j][1]
					tr0=self.trans[tr_id]
					hidden_state_num=dic_values[j][2]
					if l == frame_start[j]: # From Intermediate to anotherIntermediate of present transition matrix
						t0=tr0[ hidden_state_num  * hidden_state[j-1] + hidden_state[j] ]
					else:  # one loop of same Intermediate of present transition matrix
						t0=tr0[ hidden_state_num  * hidden_state[j] +  hidden_state[j] ]
					
					
				# Output probability (preprocessed of log10( x /(Priori probability))
				hmm_id=dic_values[j][0]
				hm0=self.hmm[hmm_id]  # hmm is list of dnn output index 
				state_index = hm0[ hidden_state[j]  ] # dnn output index
				p0=dnn0[ state_index] [ frames]
				
				# transition + Output probability (multiply becomes log-add)
				n_score[j]+=(t0 + p0)
				am_score+=(t0 + p0)
				
				# count up one frame for next
				frames+=1
			
			#normalized by number of frames
			n_score[j] /=( (frame_end[j] - frame_start[j]) + 1)
			
		return n_score, am_score
	
	
	
	def find_key_and_values(self, name0):
		dic_values=[]
		for x in (name0):
			dic_value=self.dic.get( x )
			if dic_value is None:
				print ('warning: no key found, return [-1 -1 -1]', x)
				dic_values.append([-1, -1, -1])
			else:
				dic_values.append(dic_value)  #[ HMM model index, transition matrix index, hidden state number]
		
		return dic_values
	
	def npload(self,IN_FILE):
		if not os.path.exists(IN_FILE):
			print ('-error: no file ', IN_FILE )
			sys.exit()
		else:
			print ('+load ', IN_FILE)
			return np.load(IN_FILE)
	
	def jload(self,IN_FILE):
		if not os.path.exists(IN_FILE):
			print ('-error: no file ', IN_FILE )
			sys.exit()
		else:
			print ('+load ', IN_FILE)
			fp=open(IN_FILE,'r')
			dic=json.load(fp)
			fp.close()
			return dic


if __name__ == '__main__':



	#Following is state aligment result example of 'dnn_output_sample0.npy' as 'はい。'
	state_alignment_result0= [
	# [start frame number, end frame number], 'physical triphone',  hidden state number ]
	[ [0  , 0  ], 'sp_S',           1 ],    # sp: silent
	[ [1  , 2  ], 'sp_S',           2 ],
	[ [3  , 3  ], 'sp_S',           5 ],
	[ [4  , 4  ], 'sp-h_B+a:_B',    1 ],    # x_B: Begin
	[ [5  , 5  ], 'sp-h_B+a:_B',    2 ],
	[ [6  , 6  ], 'sp-h_B+a:_B',    3 ],
	[ [7  , 13 ], 'h_B-a_I+i_E',    1 ],    # x_E: End
	[ [14 , 14 ], 'h_B-a_I+i_E',    2 ],
	[ [15 , 17 ], 'h_B-a_I+i_E',    3 ],
	[ [18 , 30 ], 'a_I-i_E+sp',     1 ],    # x_I: Intermediate
	[ [31 , 32 ], 'a_I-i_E+sp',     2 ],
	[ [33 , 43 ], 'a_I-i_E+sp',     3 ],
	[ [44 , 44 ], 'sp_S',           1 ],
	[ [45 , 71 ], 'sp_S',           2 ],
	[ [72 , 76 ], 'sp_S',           4 ],
	[ [77 , 94 ], 'sp_S',           2 ],
	[ [95 , 95 ], 'sp_S',           5 ],
	]
	# load dnn output
	dnnout0=np.load('dnn_output_sample0.npy') 
	
	# ... data preparation was completed 
	
	# class Class_calcu instance
	cal0= Class_calcu()
	
	# require Log-likelihood (n_score and AM score)
	n_score, am_score= cal0.calcu(state_alignment_result0, dnnout0)
	
	# print Log-likelihood
	print ("[[start frame number, end frame number], 'physical triphone',  hidden state number]  n_score(Log-likelihood)")
	for i in range (len(state_alignment_result0)):
		print ( state_alignment_result0[i], n_score[i])
		
	print (' am_score: ', am_score)



# This file uses TAB.
