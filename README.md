# �����M����DNN-HMM�̑ΐ��ޓx�̌v�Z

## �T�v  

�����F���G���W��Julius�̃f�B�N�e�[�V�����L�b�g�Ɋ܂܂��DNN-HMM���f���𗘗p���đΐ��ޓx���v�Z����python������Ă݂��B   
  
## �g����  
### 1.model�̏���  
Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4���_�E�����[�h����B<http://julius.osdn.jp/index.php?q=dictation-kit.html>  
model/�ȉ���W�J����B 

  
### 2.�e�v���O�����̐���  
  
- read_binhmm.py  Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f����ǂݍ���,HMM���f��,�J�ڍs��,Triphone�Ƃ̑Ή����o�͂���N���X�B
- calu_likelihood.py�@���m�́ATriphone�Ƃ���HMM�̏�Ԃ̎��n�񂩂�ΐ��ޓx���v�Z����N���X�B  
  
  
```
python calu_likelihood.py
```
�ɂ��ΐ��ޓx�̌v�Z��ł��B  
![�ΐ��ޓx�̌v�Z��](Log_likelihood_calcuation_result0.png)  




### 3.�e�f�[�^�̐���  
�ȉ��̃f�[�^�́AJulius�f�B�N�e�[�V�����L�b�gversion 4.4�Ɋ܂܂��f�[�^�𗘗p���č쐬���Ă��܂��B  
  
- hmm_model_data0.npy�@Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f����������HMM���f��(DNN�̏o�͂ւ̃C���f�b�N�X�j
- transition_matrix_data0.npy Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f�����������J�ڍs��
- dictionary_hmm_trans_hiddenstatenum0.json  Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f������������L��Triphone�Ƃ̑Ή�
```
python read_binhmm.py
``` 
�ō쐬�ł��܂��B   
 
- PLAY-16.wav�@�����g�`�̃T���v��
- dnn_output_sample0.npy�@DNN�̏o�͂̃T���v��


  

## ����  
DNN�̏o�͂̋��ߕ��ɂ��Ă�[Wave-DNN](https://github.com/shun60s/Wave-DNN) ���Q�Ƃ��Ă��������B  

## ���C�Z���X   
�ȉ��̃��C�Z���X�����Q�Ƃ̂��ƁB   
LICENSE-Julius Dictation Kit.txt  
LICENSE-Julius.txt  






