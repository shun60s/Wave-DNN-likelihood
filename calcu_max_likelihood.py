# coding=utf-8

###############################################################################
#Description:
# calculate maximum Log-likelihood of DNN-HMM model based on Viterbi algorithm
#
#
# Input:
#    time series of triphone
#    time series of dnn output
#
#    hmm_model
#    transition_matrix
#    dictionary_hmm_transition-matrix_hidden-state
#
# Output:
#     maximum Log-likelihood (am_score)
#
#############################################################################
# This use Julius dictation-kit-v4.4 data
# Please see LICENSE-Julius Dictation Kit.txt
#   
#Date 2018-04-03
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


class Class_calcu_max_likelihood(object):
	def __init__(self,):
		self.hmm=self.npload('hmm_model_data0.npy')
		self.trans=self.npload('transition_matrix_data0.npy')
		self.dic=self.jload('dictionary_hmm_trans_hiddenstatenum0.json')
		print ('  ') # write space
		self.low_limit= -1.0E+6  # set as log10(0), minus infinity
		
	
	def calcu_max(self, candidates, dnnout):
		# check dnn output matrix shape
		if dnnout.shape[0] != 4874 :
			print('-warning: dnnout.shape[0] != 4874, matrix was transposed ', dnnout.shape[0])
			dnn0=dnnout.T
		else:
			dnn0=dnnout
		
		results={}
		for x in (candidates.keys()):
			strans, shmm, slist= self.serize_hmm(candidates[x])
			
			# initial state should be 1st state of 1st triphone
			am_scores=np.ones( len(shmm), dtype=np.float32) * self.low_limit  # reset to lowest value
			am_scores[0]= dnnout[shmm[0]][0]
			out_probs=np.zeros(len(shmm), dtype=np.float32)
			
			state_paths=np.ones( [dnn0.shape[1], len(shmm)], dtype=np.int) * -1 # reset to -1
			# state_path: 
			#   index is present state
			#   value is the state of one step before 
			
			
			for loop in range (1 , dnn0.shape[1]):
				out_probs=[ dnnout[shmm[i]][loop] for i in range (len(shmm)) ]
				state_paths[loop]=np.argmax(strans + am_scores, axis=1) # set  prvious state
				am_scores=np.max(strans + am_scores, axis=1) + out_probs
				#state_paths[loop][np.where( (np.max(strans + am_scores, axis=1)) <= self.low_limit+100 )]=-1 # mark the state ,if value <= low_limit+alfa
				#am_scores[am_scores < self.low_limit]=self.low_limit # reduce to low limit, if underflow
				
				
			# finally should be terminated at last triphone
			last_index=np.argmax(strans[-1] + am_scores)
			am_scores_include_final=np.max(strans[-1] + am_scores)
			
			# trace states
			flist0=self.back_trace(state_paths, last_index, slist)
			
			# store am_scores[last_index] as maximum Log-likelihood
			results[x]= am_scores[last_index]
			
			# print out
			if True :
				print (conv2shtjis(x))  # word, windows use shift-jis as char code
				print ('am_score= ', am_scores[last_index], '  am_score_include_final= ', am_scores_include_final)
				print ('trace list..')
				for i in range (len(flist0)):
					print (flist0[i])
				print ('  ') # write space
			
		return results
	
	
	def back_trace(self,state_paths,last_index, slist):
		index0=last_index
		rlist=[ slist[last_index] ]
		for j in range ( state_paths.shape[0]-1, 0 ,-1):
			index0= state_paths[j][index0]
			rlist.append( slist[index0]) 
		# reverse back trace list
		flist=list(reversed(rlist))
		
		word0=flist[0][0]
		sp=0
		ep=0
		flist0=[]
		for j in range (1,len(flist)):
			if flist[j][0] == word0:
				ep=j
			else:
				flist0.append( [ word0, sp, ep])
				word0 =flist[j][0]
				sp=j
				ep=j
		flist0.append( [ word0, sp, ep])
		word0 =flist[j][0]
		
		return flist0  # =[[triphone, start frame number, end frame number], ...]
	
	
	def  serize_hmm(self, triphones):
		dic_values=self.find_key_and_values( triphones )
		len0= np.sum(dic_values, axis=0)[2]  # sum of every hidden states number
		len0 = len0  - (len(triphones) * 2) + 1   # remove initial state, and   terminal state and next inital state is overlapped
		serize_trans = np.ones( (len0,len0), dtype=np.float32) * self.low_limit  # reset lowest value
		serize_hmm =   np.zeros( len0, dtype=np.int) #last is terminal state
		serize_list=[] # list of serized triphone and state
		
		c0=0 # counter
		c1=0 # counter
		for loop in range (len(triphones)):
			hmm0=self.hmm[ dic_values[loop][0] ]
			tr0=self.trans[ dic_values[loop][1] ]
			hidden_state_num=dic_values[loop][2]
			
			for i in range (1,(hidden_state_num-1)):
				serize_hmm[c0]=hmm0[i]
				c0+=1
				serize_list.append([triphones[loop], i])
				for j in range (1,hidden_state_num):
					serize_trans[ c1 + i - 1][ c1 + j -1]=tr0[ hidden_state_num * i + j ]
			c1+=(hidden_state_num-2)  # since remove initial state and overlapped state
		
		return serize_trans.T , serize_hmm, serize_list  # serize_trans was inversed as .T
	
	
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
			
			
def conv2shtjis(list0):
	# convert from UTF-8 to shift-JIS
	if sys.version_info.major == 2 and  os.name == 'nt' :
		return  (list0.decode('utf-8')).encode('shift-jis')  # python 2.xx on windows
	else:
		return list0  # nothing done


if __name__ == '__main__':

	# Regarding to convert from word to phoneme, see model/lang_m/bccwj.60k.pdp.htkdic
	
	# time series of triphone
	candidates_logical_triphone={
	'はい' :   [ 'sp_S', 'sp_S-h_B+a_I', 'h_B-a_I+i_E', 'a_I-i_E+sp_S',   'sp_S'],
	'プレイ' : [ 'sp_S', 'sp-p_B+u_I',  'p_B-u_I+r_I', 'u_I-r_I+e:_E', 'r_I-e:_E+sp', 'sp_S' ],
	'第' :     [ 'sp_S', 'sp-d_B+a_I',  'd_B-a_I+i_E', 'a_I-i_E+sp', 'sp_S' ]
	}
	
	# Regarding to convert from logical_triphone to physical_triphone, see model/dnn/logicalTri
	candidates_physical_triphone={
	'はい' :   [ 'sp_S', 'sp-h_B+a:_B', 'h_B-a_I+i_E', 'a_I-i_E+sp',   'sp_S'],
	# sp_S-h_B+a_I sp-h_B+a:_B
	# a_I-i_E+sp_S a_I-i_E+sp
	'プレイ' : [ 'sp_S', 'sp-p_B+N_B',  'ky_B-u_I+by_B', 'spn-r_I+e:_B', 'by_B-e:_E+sp', 'sp_S' ], 
	# sp-p_B+u_I sp-p_B+N_B  
	# p_B-u_I+r_I ky_B-u_I+by_B
	# u_I-r_I+e:_E spn-r_I+e:_B
	# r_I-e:_E+sp by_B-e:_E+sp
	'第' :     [ 'sp_S', 'sp-d_B+a:_B',  'd_B-a_I+i:_E', 'a_I-i_E+sp', 'sp_S' ]
	# sp-d_B+a_I sp-d_B+a:_B
	# d_B-a_I+i_E d_B-a_I+i:_E
	}
	
	# load dnn output
	dnnout0=np.load('dnn_output_sample0.npy') 
	# ... data preparation was completed 
	
	
	# class Class_calcu_maxlh instance
	cal0= Class_calcu_max_likelihood()
	
	# get maximum Log-likelihood per candidate
	result0=cal0.calcu_max(candidates_physical_triphone, dnnout0 )
	print ('result',result0)



# This file uses TAB.
