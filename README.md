# 音声信号のDNN-HMMの対数尤度の計算

## 概要  

音声認識エンジンJuliusのディクテーションキットに含まれるDNN-HMMモデルを利用して対数尤度を計算するpythonを作ってみた。   
  
## 使い方  
### 1.modelの準備  
Juliusのディクテーションキットversion 4.4をダウンロードする。<http://julius.osdn.jp/index.php?q=dictation-kit.html>  
model/以下を展開する。 

  
### 2.各プログラムの説明  
  
- read_binhmm.py  Juliusのディクテーションキットversion 4.4のDNN-HMM音響モデルを読み込み,HMMモデル,遷移行列,Triphoneとの対応を出力するクラス。
- calu_likelihood.py　既知の、TriphoneとそのHMMの状態の時系列から対数尤度を計算するクラス。  
  
  
```
python calu_likelihood.py
```
による対数尤度の計算例です。  
![対数尤度の計算例](Log_likelihood_calcuation_result0.png)  




### 3.各データの説明  
以下のデータは、Juliusディクテーションキットversion 4.4に含まれるデータを利用して作成しています。  
  
- hmm_model_data0.npy　Juliusのディクテーションキットversion 4.4のDNN-HMM音響モデルから作ったHMMモデル(DNNの出力へのインデックス）
- transition_matrix_data0.npy Juliusのディクテーションキットversion 4.4のDNN-HMM音響モデルから作った遷移行列
- dictionary_hmm_trans_hiddenstatenum0.json  Juliusのディクテーションキットversion 4.4のDNN-HMM音響モデルから作った上記とTriphoneとの対応
```
python read_binhmm.py
``` 
で作成できます。   
 
- PLAY-16.wav　音声波形のサンプル
- dnn_output_sample0.npy　DNNの出力のサンプル


  

## 注意  
DNNの出力の求め方については[Wave-DNN](https://github.com/shun60s/Wave-DNN) を参照してください。  

## ライセンス   
以下のライセンス文を参照のこと。   
LICENSE-Julius Dictation Kit.txt  
LICENSE-Julius.txt  






