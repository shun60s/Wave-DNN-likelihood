# coding=utf-8

###############################################################################
#Description:
# read Julius dictation-kit-v4.4 model/dnn/binhmm.SID
#
# output:
#       'instance name'.HMM( 42186, 7)  HMM model data (DNN output index), numpy array(dtype=int)
#       'instance name'.trans( 42186, 49)  transition matrix data, numpy array(dtype=float32)
#       'instance name'.dic  python's dictionary[key='physical triphone'] = [ HMM model index, transition matrix index, hidden state number]
#
# If SAVE_FLAG is True, then save output as npy files and json file
#
# warning: this is not complete compatible with julius.
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
import struct
import numpy as np
import json

# Check version
# Windows 10 (64bit)
# Python 2.7.12 on win32 (Windows version)
# numpy (1.14.0)


# information of Julius dictation-kit-v4.4 model/dnn/binhmm.SID
#
# State data number (= DNN output number) 4874
# HMM model number (= transition matrix number) 42186
#    transition matrix:
#        Size (hidden state num x hidden state num) is 5x5 or 7x7
#        Value was log10?  ex: 1-> log10(1)=0,  0->log10(0) converted to -1.000000 as lowe limit


class Class_readbinhmm(object):
	def __init__(self, IN_FILE='model/dnn/binhmm.SID', SAVE_FLAG=False):
		if not os.path.exists(IN_FILE):
			print ('-error: no file ', IN_FILE )
			sys.exit()
			
		self.f=open(IN_FILE,"rb")
		print ('+open file of ', IN_FILE)
		
		buf=self.f.read(9) #read version
		if buf == 'JBINHMMV2':
			print ('+version ', buf)
		else:
			print ('-error: version mis-match ', buf)
			sys.exit()
		
		self.f.read(1) #read one char EOF 0x00 
		
		buf=self.f.read(2) #read qualifier string
		if buf == '_V':
			self.inv=True
			print ('+VARINV ', buf)
		else:
			print ('-error: other than VARINV ', buf)
			sys.exit()
		
		self.f.read(1)  #0x00
		# skip... acoustic analysis parameter
		# self.f.read(64)
		
		
		# option
		self.stream_info_num= self.get_short()  # Julius supports only single stream
		print ('+stream_info_num ', self.stream_info_num) # 1
		self.maxstreamnum=50
		self.stream_info_vsize= self.get_short()
		for i in range (self.maxstreamnum-1):
			self.get_short()  # read maxstreamnum counts as whole
		self.vec_size= self.get_short()
		print ('+vec_size ', self.vec_size)  # 40
		self.cov_type= self.get_short()
		print ('+cov_type ', self.cov_type)  # only C_INV_DIAG is supported in Julius
		self.dur_type= self.get_short()
		print ('+dur_type ', self.dur_type)  # No duration model is supported in Julius
		self.param_type= self.get_short()
		print ('+param_type ', self.param_type)  # 9=F_USER,
		
		# type of mixture tying
		self.is_tied_mixture=self.get_unsigned_char()
		print ('+is_tied_mixture ', self.is_tied_mixture)  # 0
		self.maxmixturenum=self.get_int()  # int
		print ('+maxmixturenum ', self.maxmixturenum)  # 98
		
		
		# transition matrix data
		self.tr_num=self.get_int()  # unsigned int
		print ('+transition matrix data number ', self.tr_num)  # 42176
		
		# allocate transition matrix data for numpy
		self.max_hidden_state_num=7  # max of hidden_state_num: almost hidden_state_num is 5, but, some is 7
		self.trans=np.zeros( (self.tr_num, self.max_hidden_state_num * self.max_hidden_state_num), dtype=np.float32)
		
		
		#  model/dnn/binhmm.SID's transition data was already log10()ed ???
		self.trans_dic={}
		for loop in range (self.tr_num):
			name0= self.get_name()
			hidden_state_num=self.get_short()

			if False: #True: 
				print ('no. ', loop)
				print ('+name ', name0)
				print ('+hidden_state_num ', hidden_state_num)
			if hidden_state_num != 5:
				print ('-warning: hidden state num is not 5, but, ', hidden_state_num, ' no. ', loop, ' name ', name0)
			if hidden_state_num > self.max_hidden_state_num:
				print ('-error: hidden state num is than self.max_hidden_state_num ', hidden_state_num, ' no. ', loop, ' name ', name0)
				sys.exit()
			
			for i in range ( hidden_state_num * hidden_state_num ):
				v0=self.get_float()
				self.trans[loop][i]=v0
				#print ( v0)
			
			# register in dictionary, id and state number (5 or 7)
			self.trans_dic[ name0[6:] ]= [ loop, hidden_state_num ]  # exclude 'trans_', 6 letters
			
			
		# variance
		self.load_variance()
		
		# mixture densities
		self.load_densities()
		
		# skip... stream weight data
		# skip... tmix data
		# skip... mixture pdf data
		
		# state data
		self.load_states()

		# HMM models
		self.md_num=self.get_int()  # unsigned int
		print ('+HMM models number ', self.md_num)  # 42176
		
		# allocate HMM model data for numpy
		self.hmm=np.ones( (self.md_num, self.max_hidden_state_num), dtype=np.int) * -1

		self.dic={}
		for loop in range (self.md_num):
			name0= self.get_name()
			hidden_state_num=self.get_short()
			
			if hidden_state_num > self.max_hidden_state_num:
				print ('-error: state num is than self.max_hidden_state_num ', hidden_state_num, ' no. ', loop, ' name ', name0)
				sys.exit()
			
			if False: #True: 
				print ('no. ', loop)
				print ('+name ', name0)
				print ('+hidden_state_num ', hidden_state_num)
			
			for i in range ( hidden_state_num ):
				v0=self.get_int()  # sid, state id
				if v0 > self.state_num:
					#print ('-warning: state index is over than state_num.  ', v0, name0)
					v0=0 # reset, is this OK?
				
				self.hmm[loop][i]=v0
				#print ('state id', v0)
			
			tr_id=self.get_int()  # tr
			#print ( 'transition id', v0)
			
			# check conflict between trans_dic
			dic_value=self.trans_dic.get( name0)
			if dic_value is None:
				print ('-error: there is no transition matrix data in trans_dic ')
				sys.exit()
			else:
				if dic_value[0] != tr_id  or dic_value[1] != hidden_state_num :
					print ('-error: mismatch between in trans_dic ')
					sys.exit()
				else:
					#print ('trans_dic[key=name]', dic_value )  # dic_value=[id , hidden_state_num ]
					self.dic[ name0 ]= [ loop, dic_value[0], hidden_state_num ] # =[ hmm_id, trans_id, hidden_state_num]
					pass
		
		# this is for just look at latest data
		if False: #True: 
			print ('last no. ', loop)
			print ('+name ', name0)
			print ('+hidden_state_num ', hidden_state_num)
			print ('trans[self.tr_num-1]', self.trans[self.tr_num-1])
			print ('hmm[self.md_num-1]', self.hmm[self.md_num-1])
			print ('dic[key=name]', self.dic[ name0] )
		
		# save as file if SAVE_FLAG is set True
		if SAVE_FLAG:
			np.save('hmm_model_data0.npy', self.hmm)  # 1154KB
			print('+save hmm_model_data0.npy')
			np.save('transition_matrix_data0.npy', self.trans)  # 8073KB
			print('+save transition_matrix_data0.npy')
			fp=open('dictionary_hmm_trans_hiddenstatenum0.json','w')  # 1412KB
			json.dump(self.dic,fp)
			fp.close()
			print('+save dictionary_hmm_trans_hiddenstatenum0.json')
		
		
	def get_unsigned_char(self,): 
		buf=self.f.read(1)
		val=struct.unpack('B',buf)  #B unsigned char
		return int(val[0])		

	def get_short(self,): 
		buf=self.f.read(2)
		val=struct.unpack('>h',buf)  #Big Endian, h short
		return int(val[0])
		
	def get_unsigned_short(self,): 
		buf=self.f.read(2)
		val=struct.unpack('>H',buf)  #Big Endian, H short
		return int(val[0])

	def get_int(self,): 
		buf=self.f.read(4)
		val=struct.unpack('>l',buf)  #Big Endian, l long
		return int(val[0])
		
	def get_float(self,): 
		buf=self.f.read(4)
		val=struct.unpack('>f',buf)  #Big Endian, f float
		return float(val[0])
		
	def get_name(self,max_name_length=100):
		buf2=[]
		for i in range (max_name_length):
			buf=self.f.read(1)
			if buf == chr(0):
				break
			else:
				buf2.append(buf)
		name0="".join(buf2)
		return name0
		
		
	def load_variance(self,):
		self.vr_num=self.get_int()  # unsigned int
		print ('+variance number ', self.vr_num)
		
		for loop in range (self.vr_num):
			name0= self.get_name()
			vec_len=self.get_short()

			if False: #True: 
				print ('no. ', loop)
				print ('+name ', name0)
				print ('+vec_len ', vec_len)
			if vec_len != 40:
				print ('-warning: vec_len is not 5', vec.len, ' no. ', loop, ' name ', name0)
				
			for i in range ( vec_len):
				v0=self.get_float()
				#print ( v0)
		
	def load_densities(self,):
		self.dens_num=self.get_int()  # unsigned int
		print ('+mixture densities number ', self.vr_num)
		
		for loop in range (self.dens_num):
			name0= self.get_name()
			mean_len=self.get_short()

			if False: #True: 
				print ('no. ', loop)
				print ('+name ', name0)
				print ('+mean_len ', mean_len)
			if mean_len != 40:
				print ('-warning: mean_len is not 40', mean_len, ' no. ', loop, ' name ', name0)
				
			for i in range (mean_len): # self.mean_len):
				v0=self.get_float()
				#print ( v0)
			
			vid=self.get_int()  # unsigned int
			v0=self.get_float() # LOGPROB
		
		
	def load_states(self,):
		self.state_num=self.get_int()  # unsigned int
		print ('+state data number ', self.state_num)
		
		for loop in range (self.state_num):
			name0= self.get_name()
			mix_num=self.get_short()   # >0: mixture 
			if mix_num < 0 :
				print ('-warning: mix num is less than 0, but, ', mix_num, ' no. ', loop, ' name ', name0)

			if False: #True: 
				print ('no. ', loop)
				print ('+name ', name0)
				print ('+mix_num ', mix_num)
				
			for i in range ( mix_num ):
				v0=self.get_int()
				#print ( v0)
				
			for i in range ( mix_num ):
				v0=self.get_float() # LOGPROB
				#print ( v0)
		
		
	def fclose(self,):
		self.f.close()
		

if __name__ == '__main__':


	hmm0= Class_readbinhmm(SAVE_FLAG=True)
	hmm0.fclose()
	# print ( hmm0.dic )  # a sample of 'instance name'.dic
	


# This file uses TAB.
