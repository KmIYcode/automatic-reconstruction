import glob
import cv2
import numpy as np
from collections import deque
import csv
import os

class AnalysisArea:
	def __init__(self, source_video, source_parameter, output_path):
		self.source_video = source_video#ディレクトリ(video*)のパス
		self.video_name = glob.glob(self.source_video + "/*.mp4")[0] #パスで指定されたディレクトリ内の動画のパス
		self.movie = cv2.VideoCapture(self.video_name)
		if not self.movie.isOpened():
			print("動画を読み込めていません。")

		#動画の情報
		self.fps = round(self.movie.get(cv2.CAP_PROP_FPS))
		self.width = int(self.movie.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.frame_count = int(self.movie.get(cv2.CAP_PROP_FRAME_COUNT))

		self.processing_frame = np.empty((self.height,self.width,3),dtype=np.uint8) #処理するフレーム
		self.pre_frame = deque()

		self.people_num = None
		
		self.readParameter(source_parameter)

		self.people_range = [[None] * self.people_num for i in range(self.frame_count)] #全フレームでの人物毎の座標

		self.output_path = output_path
		self.output_video_x = self.output_path + "/video{}".format(self.source_video[-1])

		#結果出力先ファイル　なければ作成
		if not os.path.exists(self.output_video_x):
			print("ディレクトリ:video{} を作成します".format(self.output_video_x))
			os.makedirs(self.output_video_x)
		
		# 指定した座標からcsvファイルを作成
		# このrectを外部から引数で与える
		self.rect = [[106, 198, 246, 457],[406, 187, 560, 440],[625, 146, 805, 455]] # rect = []2次元list
		self.createCoordinateCsv()

		# 座標を読み込み
		self.readPeopleCoordinate()

		self.area = []
		self.area_mean = []
		self.area_max = []
		self.area_sec = []
		

	def createCoordinateCsv(self):
		is_csv = os.path.isfile(self.source_video + "/coordinate.csv")
		if is_csv:
			os.remove(self.source_video + "/coordinate.csv")
		with open(self.source_video + "/coordinate.csv", "a") as f:
			writer = csv.writer(f)
			for i in range(self.frame_count):
				for j in range(len(self.rect)):
					writer.writerow([i+1,j+1] + self.rect[j])

	def readParameter(self, p_path):
		#動画中の人物の数
		with open(p_path + "/people_num.txt") as f:
			self.people_num = int(f.readlines()[0])
		f.close()


	#YOLOで取得した座標のcsvデータを読み込み
	#csvファイルは前処理の必要あり（余計な人物がいないこと、途中で人物のIDが入れ替わらないこと）
	def readPeopleCoordinate(self):
		csv_file = glob.glob(self.source_video + "/*.csv") #拡張子がcsvのファイルのリストを取得
		if not csv_file: #csvファイルがない、座標を読み込まないときの処理をここで分岐
			print("!!csvファイルが存在しません!!")
			#ここで事前に人物の座標を入力する
			# self.people_range = [[330, 463, 286, 539],[907, 446, 311, 523],[1320, 386, 432, 620]]
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





	"""面積から動作量検出"""
	#フレーム間差分
	def diffFrame(self, now_src, pre_src):
		now_src = cv2.cvtColor(now_src, cv2.COLOR_BGR2GRAY)
		pre_src = cv2.cvtColor(pre_src, cv2.COLOR_BGR2GRAY)
		diff_src = cv2.absdiff(now_src, pre_src)

		return diff_src

	#差分フレームを前処理
	def processingDiffFrame(self, diff_src):
		diff_src = cv2.medianBlur(diff_src, 5) #平滑化フィルタ
		th, diff_src = cv2.threshold(diff_src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #閾値を自動で設定
		kernel = np.ones((3, 3), np.uint8) #カーネル
		diff_src = cv2.morphologyEx(diff_src, cv2.MORPH_OPEN, kernel) #オープニング処理

		return diff_src

	#白ピクセルをカウントし面積を求める
	def calcArea(self, people_frame_src, f_num_src, DIFF):
		if f_num_src == DIFF:
			area_tmp_frame = []
			for i in range(self.people_num):
				area_tmp_frame.append(0)
			for i in range(DIFF):
				self.area.append(area_tmp_frame)
		area_tmp_frame = []
		for i in range(self.people_num):
			# print("people_range = {}".format(self.people_range[f_num_src]))
			if self.people_range[f_num_src][i] != None: #人が検知されている時
				roi = people_frame_src[self.people_range[f_num_src][i][1]:self.people_range[f_num_src][i][1]+self.people_range[f_num_src][i][3],\
										self.people_range[f_num_src][i][0]:self.people_range[f_num_src][i][0]+self.people_range[f_num_src][i][2]]
				people_area = self.people_range[f_num_src][i][3] * self.people_range[f_num_src][i][2]
				area_tmp_people = cv2.countNonZero(roi) / people_area
				area_tmp_frame.append(area_tmp_people)
			else: #人が検知されなかった時
				area_tmp_frame.append(0)
		self.area.append(area_tmp_frame)

	#差分用のpre_frameを1個ずらす
	def setFrame(self):
		self.pre_frame.popleft()
		self.pre_frame.append(self.processing_frame)#確認
	"""面積から動作量検出部終了"""


	#面積の移動平均を計算
	def meanArea(self, DIFF):
		_sum, num, _range = 0, 0, 10
		for i in range(len(self.area)):
			area_sub = []
			for p in range(self.people_num):
				_sum = 0.0
				num = 0.0
				for j in range(-_range, _range+1, 1):
					idx = i + j
					if idx >= DIFF and idx < len(self.area):
						_sum += self.area[idx][p]
						num += 1
				ave = _sum / num
				area_sub.append(ave)
			self.area_mean.append(area_sub)
		self.writeAreamean()


	#1秒間の面積を計算して保存
	def secArea(self):
		#最大値
		for i in range(self.people_num):
			self.area_max.append(0.0)
		#1秒間の面積
		for i in range(0, len(self.area_mean), self.fps):
			area_sub = []
			for p in range(self.people_num):
				area_sub.append(0.0)
				for j in range(i, i+self.fps, 1):
					if j == len(self.area_mean):
						break
					area_sub[p] += self.area_mean[j][p]
				if area_sub[p] > self.area_max[p]:
					self.area_max[p] = area_sub[p]
			self.area_sec.append(area_sub)
		self.writeAreasec()
		self.writeAreamax()


	def writeAreamean(self):
		print("---AnalysisArea : writeAreamean: {}---".format(self.source_video[-1]))
		with open(self.output_video_x + "/Areamean_video{}.csv".format(self.source_video[-1]), \
					"w", newline='') as f:
			writer = csv.writer(f)
			writer.writerows(self.area_mean)


	def writeAreasec(self):
		print("---AnalysisArea : writeAreasec: {}---".format(self.source_video[-1]))
		with open(self.output_video_x + "/Areasec_video{}.csv".format(self.source_video[-1]), \
					"w", newline='') as f:
			writer = csv.writer(f)
			writer.writerows(self.area_sec)

	def writeAreamax(self):
		print("---AnalysisArea : writeAreamax: {}---".format(self.source_video[-1]))
		with open(self.output_video_x + "/Areamax_video{}.csv".format(self.source_video[-1]), \
					"w", newline='') as f:
			writer = csv.writer(f)
			writer.writerow(self.area_max)

	def writeArea(self):
		print("---AnalysisArea : writeArea: {}---".format(self.source_video[-1]))
		with open(self.output_video_x + "/Area_video{}.csv".format(self.source_video[-1]), \
					"w", newline='') as f:
			writer = csv.writer(f)
			writer.writerows(self.area)




























