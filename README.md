# �����M����DNN-HMM�̑ΐ��ޓx�̌v�Z

## �T�v  

�����F���G���W��Julius�̃f�B�N�e�[�V�����L�b�g�Ɋ܂܂��DNN-HMM���f���𗘗p���đΐ��ޓx���v�Z����python������Ă݂��B   
  
## �g����  
### 1.model�̏���  
Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4���_�E�����[�h����B<http://julius.osdn.jp/index.php?q=dictation-kit.html>  
model/�ȉ���W�J����B 

  
### 2.�e�v���O�����̐���  
  
- read_binhmm.py  Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f����ǂݍ���,HMM���f��,�J�ڍs��,Triphone�Ƃ̑Ή����o�͂���N���X�B
- calcu_likelihood.py�@���m�́ATriphone�Ƃ���HMM�̏�Ԃ̎��n�񂩂�ΐ��ޓx���v�Z����N���X�B  
  
  
```
python calcu_likelihood.py
```
���L�͑ΐ��ޓx�̌v�Z��ł��B  
![�ΐ��ޓx�̌v�Z��](Log_likelihood_calcuation_result0.png)  
Julius�̐ݒ�Ł@-salign �i�F�����ʂ�HMM��Ԃ��Ƃ̃A���C�������g���ʂ��o�́j�@�𗘗p���� Triphone�Ƃ���HMM�̏�Ԃ̎��n��𓾂邱�Ƃ��ł��܂��B  

  
- calcu_max_likelihood.py Triphone�̎��n�񂩂�ő�ΐ��ޓx���v�Z����N���X�B  
```
python calcu_max_likelihood.py
```
���L�͍ő�ΐ��ޓx�̌v�Z��ł��B  
![�ő�ΐ��ޓx�̌v�Z��](result_maximum_likelihood.png)  
DNN�̏o�͂̃T���v���́A�u�͂��v�u��v�u�v���C�v�̂R��Triphone�̎��n��ɂ��Ă̍ő�ΐ��ޓx�iam_score=�j���v�Z���Ă��܂��B  
�u��v��27.36�ƍ����̂ł����AJulius�̔F�����ʂ͍ŏI�I�ɂ�22.069�̒P�Ɖ��́u�͂��B�v�Ɣ��肳��܂����B  
���ۂ̔��b�́u�v���C�v�́A���b��Ԃ�����(sp_S)�ɃA�T�C������A-46.86�ƒႢ�l�ɂȂ��Ă��܂��B  





### 3.�e�f�[�^�̐���  
�ȉ��̃f�[�^�́AJulius�f�B�N�e�[�V�����L�b�gversion 4.4�Ɋ܂܂��f�[�^�𗘗p���č쐬���Ă��܂��B  
  
- hmm_model_data0.npy�@Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f����������HMM���f��(DNN�̏o�͂ւ̃C���f�b�N�X�j
- transition_matrix_data0.npy Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f�����������J�ڍs��
- dictionary_hmm_trans_hiddenstatenum0.json  Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4��DNN-HMM�������f������������L��Triphone�Ƃ̑Ή�
```
python read_binhmm.py
``` 
�ō쐬�ł��܂��B   
 
- PLAY-16.wav�@�����g�`�̃T���v���B�F�����ʂ́u�͂��B�v�ɂȂ��Ă��邪�A���ۂɂ́u�v���[�v�Ɣ��b���Ă���B�Ō���̉��͏�����Ԃ̓s���㖳������Ă���B  
- dnn_output_sample0.npy�@DNN�̏o�͂̃T���v���B Julius�f�B�N�e�[�V�����L�b�gversion 4.4��DNN(dnnclient)�ł𗘗p����DNN�o�͂����o�������́B  
  
DNN�̏o�͂̋��ߕ��ɂ��Ă�[Wave-DNN](https://github.com/shun60s/Wave-DNN) ���Q�Ƃ��Ă��������B  

## ���C�Z���X   
�ȉ��̃��C�Z���X�����Q�Ƃ̂��ƁB   
LICENSE-Julius Dictation Kit.txt  
LICENSE-Julius.txt  






