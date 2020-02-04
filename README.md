# TensorFlowを使用したペンギン識別AI

# 動作環境
- Windows 10
	- Anaconda(Python 3.7)
		- TensorFlow
		- TensorFlow Hub
		- OpenCV
		- Flask
- GCE(東京, asia-northeast1-c　N2, n2-highcpu-16)
	- Ubuntu 18.04.3 LTS
		- Python 3.6.9
		- TensorFlow 1.14.0
		- TensorFlow Hub 0.7.0
		- Flask 1.1.1
		- OpenCV 4.1.2
- Ubuntu 18.04.3 LTS 日本語Remix
	- Python 3.6.9
	- TensorFlow 1.14.0
	- TensorFlow Hub 0.7.0
	- Flask 1.1.1
	- OpenCV 3.2.0

# 環境構築　※Ubuntu 18.04.3 LTSでの場合

## Python3用のpipをインストール。
```
sudo apt-get install python3-pip
```

## Flaskをインストール。
```
pip3 install flask
```

## TensorFlowをインストール。
```
pip3 install tensorflow
```

## TensorFlow Hubをインストール。
```
pip3 install tensorflow-hub
```

## OpenCVをインストール。
```
sudo apt install python3-opencv
```

# 動かし方　※Ubuntu 18.04.3 LTSでの場合
## 任意のディレクトリに移動し、リポジトリからクローンする。
```
git clone https://github.com/rockhopper-penguin/Kist.git
```

## クローンしたフォルダに移動。
```
cd Kist
```

## 実行！
```
sudo python3 index.py
```

# Windowsでの実行の際の注意点
Windowsで動作させる場合、index.pyのapp.runの所を空にする or ウェルノウンポート以外を指定します。

```
app.run(host="0.0.0.0", port=80)
```

↓

```
app.run()
```

# 参考
## <a href="https://www.tensorflow.org/hub/tutorials/image_retraining">How to Retrain an Image Classifier for New Categories</a>
## <a href="https://ensekitt.hatenablog.com/entry/2018/06/27/200000">FlaskとOpenCVで投稿された画像をOpenCVで加工して返す話</a>