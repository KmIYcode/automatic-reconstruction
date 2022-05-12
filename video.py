import glob
import cv2
import numpy as np
from collections import deque
import csv
import os

class Video:
	def __init__(self, source_video, source_parameter, output_path):
		self.source_video = source_video#ディレクトリ(video*)のパス
		self.video_name = glob.glob(self.source_video + "/*.mp4")[0] #パスで指定されたディレクトリ内の動画のパス
		self.movie = cv2.VideoCapture(self.video_name)
		if not self.movie.isOpened():
			print("動画を読み込めていません。")

		self.source_parameter = source_parameter

		self.output_path = output_path
		self.output_video_x = self.output_path + "/video{}".format(self.source_video[-1])
		 #結果出力先ファイル　なければ作成
		if not os.path.exists(self.output_video_x):
			print("ディレクトリ: {} が存在しません".format(self.output_video_x))

		#動画の情報
		self.fps = round(self.movie.get(cv2.CAP_PROP_FPS))
		self.width = int(self.movie.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.frame_count = int(self.movie.get(cv2.CAP_PROP_FRAME_COUNT))

		self.people_num = None
		self.TH = None
		self.readParameter(self.source_parameter) #再構成に必要な情報を読み込み
		print("{} : {}".format(self.video_name, self.people_num))


		self.people_range = [[None] * self.people_num for i in range(self.frame_count)] #全フレームでの人物毎の座標

		self.area = []
		self.area_mean = []
		self.area_max = []
		self.area_sec = []
		self.area_th = []

		self.processing_frame = np.empty((self.height,self.width,3),dtype=np.uint8) #処理するフレーム


	def readParameter(self, p_path):
		 #動画中の人物の数
		with open(p_path + "/people_num.txt") as f:
			self.people_num = int(f.readlines()[1])
		f.close()
		 #動作量の閾値
		with open(p_path + "/threshold.txt") as f:
			self.TH = float(f.readlines()[1])
		f.close()

	def readArea(self):
		for line in open(self.output_video_x + "/Areasec_video{}.csv".format(self.source_video[-1]), "r", encoding = "utf_8"):
			line = eval(line)
			self.area_sec.append(line)

	def readAreamax(self):
		for line in open(self.output_video_x + "/Areamax_video{}.csv".format(self.source_video[-1]), "r", encoding = "utf_8"):
			line = eval(line)
			self.area_max = line


	#YOLOで取得した座標のcsvデータを読み込み
	 #csvファイルは前処理の必要あり（余計な人物がいないこと、途中で人物のIDが入れ替わらないこと）
	def readPeopleCoordinate(self):
		csv_file = glob.glob(self.source_video + "/*.csv") #拡張子がcsvのファイルのリストを取得
		if not csv_file: #csvファイルがない、座標を読み込まないときの処理をここで分岐
			print("!!csvファイルが存在しません!!")
			#ここで事前に人物の座標を入力する
			#self.people_range = []
		elif len(csv_file) > 1:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
			print("!!複数のcsvファイルが同時に存在しています!!") #複数のcsvファイルはエラー
		else: #座標を読み込むときの処理
			print("csvファイル:{} を確認しました".format(csv_file[0]))
			with open(csv_file[0]) as f:
				reader = csv.reader(f)
				line = [row for row in reader]
			line = [list(map(int, row)) for row in line] #読み込んだcsvをint型に変換
			
			for i in range(len(line)):
				self.people_range[line[i][0]-1][line[i][1]-1] = line[i][2:]
			for i in range(len(self.people_range)): #(x_min,y_min,x_max,y_max)を(x_min,y_min,width,height)に変換
				for j in range(self.people_num):
					if self.people_range[i][j] != None:
						self.people_range[i][j][2] = self.people_range[i][j][2] - self.people_range[i][j][0]
						self.people_range[i][j][3] = self.people_range[i][j][3] - self.people_range[i][j][1]
			
	


	def setThreshold(self):
		for i in range(self.people_num):
			self.area_th.append(self.area_max[i] * self.TH)


