import glob
import cv2
import numpy as np
import csv
import os
import math
import random
from video import Video
from scipy.optimize import differential_evolution

class DE:
	def __init__(self, src_t, src_video, src_selected_people, src_selected_camera, src_count_people, src_count_camera, src_ratio):
		self.t = src_t
		self.video = src_video
		self.selected_people = src_selected_people
		self.selected_camera = src_selected_camera
		self.count_people = src_count_people
		self.count_camera = src_count_camera
		self.ratio = src_ratio
		self.fps = self.video[0].fps
		self.people_num = self.video[0].people_num

		self.source_parameter = self.video[0].source_parameter

		self.n1 = None
		self.D = None
		self.weight = None
		self.evaluation_num = 3
		self.generation = None
		self.Cr = 0.1

		self.readPrameter(self.source_parameter)

		self.section = -self.n1 + self.D #区間の大きさ(ステップ数) |n1~n2|
		self.n2 = self.D -1
		self.Np = 10 * self.D

		self.ideal_ratio = []
		self.euclid_max = 0.0
		self.sub_max = 0.0 #動作量の閾値との差の最大値

		self.Ev = [0] * self.evaluation_num

		#解の範囲
		c_max = len(self.ratio) * len(self.video)
		self.candidate_min = 0.0 #解の下限
		self.candidate_max = float(c_max) #解の上限
		self.bounds = [(self.candidate_min, self.candidate_max) for i in range(self.D)]


		self.calcIdealratio()
		self.calcAreaMax()


	def readPrameter(self, p_path):
		#評価区間の始まりを読み込み（n1）
		with open(p_path + "/n1.txt") as f:
			self.n1 = int(f.readlines()[0])
		f.close()

		#求める変数の数
		with open(p_path + "/D.txt") as f:
			self.D = int(f.readlines()[0])
		f.close()

		#重み係数の読み込み (辞書型)
		with open(p_path + "/weight.csv", "r") as f:
			reader = csv.DictReader(f)
			for row in reader:
				self.weight = row
			for key, val in self.weight.items():
				self.weight[key] = float(val)
		#世代の数
		with open(p_path + "/generation.txt") as f:
			self.generation = int(f.readlines()[0])
		f.close()


	#理想の割合と距離の最大値を計算
	def calcIdealratio(self):
		R = 0.0 #比の合計
		for i in range(len(self.ratio)):
			R += self.ratio[i]
		for i in range(len(self.ratio)):
			self.ideal_ratio.append(self.ratio[i] / R)

		#距離の最大値
		for i in range(len(self.ratio)):
			tmp_euc = 0.0
			for j in range(len(self.ratio)):
				if j == i:
					tmp = math.pow(1.0 - self.ideal_ratio[j], 2.0)
				else:
					tmp = math.pow(0.0 - self.ideal_ratio[j], 2.0)
				tmp_euc += tmp
			tmp_euc = math.sqrt(tmp_euc)
			if self.euclid_max < tmp_euc:
				self.euclid_max = tmp_euc

	#動作量閾値との差の最大値を計算
	def calcAreaMax(self):
		for v in range(len(self.video)):
			for p in range(self.people_num):
				tmp = self.video[v].area_max[p] - self.video[v].area_th[p]
				if self.sub_max < tmp:
					self.sub_max = tmp

	def evaluation_function(self, x):
		dist, deg, swi = 0, 0, 0

		if self.weight["DISTANCE"] != 0:
			dist = self.weight["DISTANCE"] * self.rateDist(x) #比率
			self.Ev[0] = dist
		if self.weight["M_DEGREE"] != 0:
			deg = self.weight["M_DEGREE"] * self.moveDegree(x) #動作量
			self.Ev[1] = deg
		if self.weight["SWITCH"] != 0:
			swi = self.weight["SWITCH"] * self.countSwitch(x) #切り替え回数
			self.Ev[2] = swi

		E = dist + deg + swi

		return E

	def newEF(self):
		result = differential_evolution(self.evaluation_function, self.bounds, maxiter=self.generation, recombination=self.Cr)
		return result, self.Ev


	#移す人物の理想の数と実際の数との距離
	def rateDist(self, x):
		count = [] #映した回数
		ideal = [] #理想の回数

		#これまで回数
		for i in range(len(self.ratio)):
			count.append(self.count_people[i])
		#評価区間内で選んだ人物を加算
		for i in range(len(self.bounds)):
			for j in range(len(self.ratio)):
				if self.toPeople(x[i]) == j:
					count[j] += 1

		#実際の割合を計算
		actual_ratio = []
		total = len(self.selected_people) + self.D #回数の合計
		for i in range(len(self.ratio)):
			actual_ratio.append(count[i] / total)

		#ユークリッド距離
		euclid = self.calcEuclid(actual_ratio, self.ideal_ratio)
		#0~1に正規化
		euclid_min = 0.0 #euclidの最小値
		euclid = self.normalize(euclid, euclid_min, self.euclid_max)

		return euclid



	def toPeople(self, src):
		# print("bestC = {}".format(src))
		people = src % len(self.ratio)
		# print("people = {}".format(people))
		# print("intpeople = {}".format(int(people)))
		return int(people)

	def toCamera(self, src):
		camera = int(src) // len(self.ratio)
		return camera

	def calcEuclid(self, actural, ideal):
		euclid = 0.0
		for i in range(len(self.ratio)):
			tmp = math.pow(actural[i] - ideal[i], 2.0)
			euclid += tmp
		euclid = math.sqrt(euclid)
		return euclid

	def normalize(self, src, minimum, maximum):
		tmp = (src - minimum) / (maximum - minimum)
		return tmp

	def moveDegree(self, x):
		sumS = 0.0 #閾値を超えた動作量の合計
		sumSmax = self.sub_max * self.section #閾値を超えた動作量の合計の最大値
		for current in range(self.n1, self.D, 1): #評価区間
			who_people = -1 #誰を注目しているか
			which_camera = -1 #どのカメラか
			p_size = len(self.selected_people)

			if current < 0: #過去
				tmp = p_size + current
				if tmp >= 0:
					who_people = self.selected_people[tmp]
					which_camera = self.selected_camera[tmp]
			else: #現在と未来
				who_people = self.toPeople(x[current])
				which_camera = self.toCamera(x[current])

			step = (self.t // self.fps) + current #現在の時刻
			#全体表示と最初の部分とパンを除く、かつ、はみ出し防止
			if who_people >= 0 and who_people < self.people_num and step < len(self.video[which_camera].area_sec):
				#閾値処理
				th = self.video[which_camera].area_th[who_people] #閾値
				S = self.video[which_camera].area_sec[step][who_people] #動作量
				subS = S -th
				if subS > 0:
					sumS += subS

		#0~1に正規化
		if sumS > 0.0:
			sumS = self.normalize(sumS, 0.0, sumSmax)
		return -sumS

	def countSwitch(self, x):
		swi = 0.0

		#判定するpの配列
		P = []
		C = []
		tmp = len(self.selected_people) + self.n1 #さかのぼる範囲
		if tmp < 0:
			tmp = 0
		for i in range(tmp, len(self.selected_people), 1): #過去
			P.append(self.selected_people[i])
			C.append(self.selected_camera[i])
		for i in range(self.D): #現在と未来
			p = self.toPeople(x[i])
			c = self.toCamera(x[i])
			P.append(p)
			C.append(c)

		#切り替え回数をカウント
		for i in range(1, len(P)):
			if P[i] != P[i - 1] or C[i] != C[i - 1]:
				swi += 1

		#正規化
		swi_max = float(self.section) - 1.0
		swi_min = 0.0
		swi = self.normalize(swi, swi_min, swi_max)
		return swi