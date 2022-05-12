import glob
import cv2
import numpy as np
import csv
import os
from video import Video
from DE import DE

class VideoReconstruct:
	def __init__(self, source_video, source_parameter, output_path):
		self.video = source_video
		self.source_parameter = source_parameter
		self.output_path = output_path

		self.video_num = len(self.video)
		self.people_num = None
		self.ratio = []

		self.fps = self.video[0].fps
		self.width = self.video[0].width
		self.height = self.video[0].height

		# if os.path.exists(self.source_video[0]):
		# 	print("ディレクトリ:{} が存在します".format(self.video[0]))

		self.frame_count_min = None

		#DE
		self.people = None #演奏者の選択
		self.camera = None #カメラの選択
		self.selected_people = [] #過去のpeopleを記録
		self.selected_camera = [] #過去のcameraを記録
		self.count_people = [] #過去のpeopleの合計を記録
		self.count_camera = [] #過去のcameraの合計を記録
		self.Ev = [] #評価値 double型の一次元配列
		self.L = None #拡大率
		self.evaluation_num = None

		self.readParameter(source_parameter) #再構成に必要な情報を読み込み
		self.initialization() #どの人物、どのカメラを使用したかの記録するためのデータを初期化

		#動画の保存
		self.codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		self.isColor = True
		self.wirter = cv2.VideoWriter(self.output_path + "/output.mp4", self.codec, self.fps, (self.width, self.height))





	def readParameter(self, p_path):
		print("---VideoReconstruct : readParameter---")
		 #動画中の人物の数
		with open(p_path + "/people_num.txt") as f:
			self.people_num = int(f.readlines()[1])
		f.close()

		 #人物を映し出す比率
		with open(p_path + "/ratio.txt") as f:
			next(f)
			for line in f:
				self.ratio.append(int(line.strip("\n")))
			print(self.ratio)

		 #拡大率
		with open(p_path + "/L.txt") as f:
			self.L = int(f.readlines()[1])
		f.close()

		 #評価項目の数
		with open(p_path + "/evaluation_num.txt") as f:
			self.evaluation_num = int(f.readlines()[1])
		f.close()


	def initialization(self):
		for i in range(len(self.ratio)):
			self.count_people.append(0)
		for i in range(self.video_num):
			self.count_camera.append(0)


	def readAnalysisAreaData(self):
		print("---VideoReconstruct : readAnalysisAreaData---")
		tmp = 2147483647 #INT_MAX
		for i in range(self.video_num):
			self.video[i].readArea();
			self.video[i].readAreamax()
			self.video[i].readPeopleCoordinate() #人物の座標情報を読み込み
			if self.video[i].frame_count < tmp:
				tmp = self.video[i].frame_count
			self.video[i].setThreshold()
		self.frame_count_min = tmp


	def changeFrameReconst(self):
		for i in range(self.video_num):
			ret, self.video[i].processing_frame = self.video[i].movie.read()


	def reconstruct(self, t):
		frameWrite = np.empty((self.height,self.width,3),dtype=np.uint8)

		 #fps毎(1秒毎)に注目領域を決定
		if t % self.fps == 0:
			self.runDE(t)

		#注目領域の適用
		if self.people != self.people_num and self.video[self.camera].people_range[t][self.people] != None:
			people_center_x = self.video[self.camera].people_range[t][self.people][0]+(self.video[self.camera].people_range[t][self.people][2]//2)
			people_center_y = self.video[self.camera].people_range[t][self.people][1]+(self.video[self.camera].people_range[t][self.people][3]//2)
			frame_reconstruct = self.video[self.camera].processing_frame.copy()
			expanding_range = self.setRect(people_center_x, people_center_y) #拡大後の範囲
			frameWrite = self.doTrimming(frame_reconstruct, expanding_range) #トリミング
		else: #全体表示
			frameWrite = self.video[self.camera].processing_frame.copy()
		return frameWrite

	def setRect(self, x, y):
		width_p = self.width // self.L
		height_p = self.height // self.L

		x_tri, y_tri = self.fixTri(x, y, width_p, height_p) #左上の座標に変換
		return (x_tri, y_tri, width_p, height_p)

	def fixTri(self, x, y, w, h):
		x_tri = x - w // 2
		y_tri = y - h // 2
		#はみ出したら修正
		if x_tri < 0:
			x_tri = 0
		if x_tri + w > self.width:
			x_tri = self.width - w
		if y_tri < 0:
			y_tri = 0
		if y_tri + h > self.height:
			y_tri = self.height - h
		return x_tri, y_tri

	def doTrimming(self, src_frame, src_range):
		frame_trim = src_frame[src_range[1]:src_range[1]+src_range[3],src_range[0]:src_range[0]+src_range[2]]
		frame_trim = cv2.resize(frame_trim, (self.width, self.height))
		return frame_trim


	def runDE(self, t):
		print("---runDE---")
		de = DE(t, self.video, self.selected_people, self.selected_camera, self.count_people, self.count_camera, self.ratio)

		result = de.newEF()
		self.Ev.append(result.fun)
		print(result)

		 #人物とカメラに変更
		self.people = de.toPeople(result.x[0]) #選択した人物
		self.camera = de.toCamera(result.x[0]) #選択したカメラ
		self.savePeopleCamera()

		#パラメータ保存用 DE時に使ったパラメータを一回だけ保存
		if t == 0:
			self.writeDifferentialEvolutionValue(de)


	def savePeopleCamera(self):
		#保存
		self.selected_people.append(self.people)
		self.selected_camera.append(self.camera)

		#カウント
		for i in range(len(self.ratio)):
			if self.people == i:
				self.count_people[i] += 1
		for i in range(self.video_num):
			if self.camera == i:
				self.count_camera[i] += 1


	def writeData(self):
		self.writeEvaluationValue()
		self.writePeopleCamera()
		self.writeVariousValue()


	def writeEvaluationValue(self):
		print("---VideoReconstruct : writeEvaluationValue---")
		with open(self.output_path + "/evaluation_value.txt", mode='w') as f:
			f.write("t")
			for i in range(self.evaluation_num):
				f.write(", E{}".format(i+1))
				if i == self.evaluation_num-1:
					f.write(", E\n")
			for i in range(len(self.Ev)):
				f.write("{}".format(i+1))
				f.write(", {}\n".format(self.Ev[i]))
		f.close()

	def writePeopleCamera(self):
		print("---VideoReconstruct : writePeopleCamera---")
		with open(self.output_path + "/selected_people_camera.txt", mode='w') as f:
			f.write("t, camera, people\n")
			for i in range(len(self.selected_people)):
				f.write("{}, {}, {}\n".format(i+1, self.selected_camera[i], self.selected_people[i]))
		f.close()

	def writeVariousValue(self):
		print("---VideoReconstruct : writeVariousValue---")
		with open(self.output_path + "/reconst_info.txt", mode='w') as f:
			f.write("人数 = {}\n".format(self.people_num))
			f.write("動画数 = {}\n".format(self.video_num))
			f.write("指定比率 = {}\n".format(self.ratio))
			# f.write("実行環境 = " + platform.system() + "\n")

	def writeDifferentialEvolutionValue(self, de):
		print("---VideoReconstruct : writeDEValue---")
		with open(self.output_path + "/DE_parameter.txt", mode='w') as f2:
			f2.write("---DEパラメータ---\n")
			f2.write("世代数:generation = {}\n".format(de.generation))
			f2.write("交叉率:Cr = {}\n".format(de.Cr))
			f2.write("---評価関数---\n")
			f2.write("n1 = {}\n".format(de.n1))
			f2.write("D = {}\n".format(de.D))
			f2.write("--> 評価区間:section = {}~{}\n".format(de.n1, de.D))
			f2.write("項の数:evaluation_num = {}\n".format(de.evaluation_num))
			f2.write("重み:weight = {}\n".format(de.weight))

	# def analysisResult(self):
	# 	self.analysisRatio()


	# def analysisRatio(self):
	# 	de_for_analysis_result = DE(t, self.video, self.selected_people, self.selected_camera, self.count_people, self.count_camera, self.ratio)



	def saveVideo(self, fW, t, tri):
		print("---VideoReconstruct : saveVideo---")
		s = len(fW)

		for i in range(s):
			self.wirter.write(fW[i])
