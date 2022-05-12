from analysis_area import AnalysisArea
from video_reconst import VideoReconstruct
from video import Video
import glob
import time
import requests
import os


source_path = "./source_fullvideo"
parameter_path = source_path + "/parameter" #./sourceフォルダ内のparemeterフォルダへのパス
output_path = "./outputs8"

print("source_path : {}".format(source_path))

DIFF = None

Flag_area = None
Flag_coordinate = None

TRIALS = None

# LINEに通知する関数
def line_notify(message):
	line_notify_token = 'oODfnr9ZRYe1JfUnw0sY0Frxi2kb7ejzkM6XO0EDWiJ'
	line_notify_api = 'https://notify-api.line.me/api/notify'
	payload = {'message': message}
	headers = {'Authorization': 'Bearer ' + line_notify_token} 
	requests.post(line_notify_api, data=payload, headers=headers)

def readParameter():
	global DIFF,Flag_area,Flag_coordinate,TRIALS
	
	#DIFF
	with open(parameter_path + "/DIFF.txt") as f:
		DIFF = int(f.readlines()[2])
	f.close()

	#Flag_area Flag_coordinate
	with open(parameter_path + "/Flag_area_coordinate.txt") as f:
		data = f.readlines()
		Flag_area = int(data[2].rstrip())
		Flag_coordinate = int(data[4])
	f.close()

	#TRIALS
	with open(parameter_path + "/TRIALS.txt") as f:
		TRIALS = int(f.readlines()[2])
	f.close()


def getVideoDirPath():#動画等保存されているフォルダのパスを取得
	video_dir_path = glob.glob(source_path + "/video?")
	video_dir_path.sort()#フォルダネーム順にソート
	print("ビデオ数: {}".format(len(video_dir_path)))
	return video_dir_path



def analysisArea(): #面積で動作量を求める
	video_dir_path = getVideoDirPath()

	anal = [AnalysisArea(v_path,parameter_path,output_path) for v_path in video_dir_path]

	f_num = 0 #フレームナンバー
	for i in range(len(video_dir_path)):
		"""解析開始"""
		while (anal[i].movie.isOpened()):
			ret, anal[i].processing_frame = anal[i].movie.read()
			if not ret:
				break
			
			"""面積で動作量を求める"""
			if f_num < DIFF:
				# anal[i].pre_frame[DIFF-1-t] = anal[i].processing_frame.copy()
				anal[i].pre_frame.append(anal[i].processing_frame)
			else:
				# diff_frame = anal[i].diffFrame(anal[i].processing_frame, anal.pre_frame[DIFF-1])
				diff_frame = anal[i].diffFrame(anal[i].processing_frame, anal[i].pre_frame[0])
				people_frame = anal[i].processingDiffFrame(diff_frame)
				anal[i].calcArea(people_frame, f_num, DIFF)

				anal[i].setFrame()

			#進捗確認
			if anal[i].frame_count <= 3000:
				if f_num % 50 == 0:
					print("AnalysisArea: video:{}  {}/{}".format(i+1, f_num, anal[i].frame_count))
			else:
				if f_num % 100 == 0:
					print("AnalysisArea: video:{}  {}/{}".format(i+1, f_num, anal[i].frame_count))
			f_num += 1 #フレームナンバーインクリメント
		anal[i].movie.release() #最後まで読み込んだ動画を開放
		f_num = 0 #次のvideoの処理をするために0にする
		"""解析終了"""
		
		"""解析データの書き出し"""
		anal[i].meanArea(DIFF)
		anal[i].secArea()
		anal[i].writeArea()




def videoReconstruct(): #動画像の再構成を行う
	video_dir_path = getVideoDirPath()

	for trial in range(1, TRIALS+1, 1):
		#動画を読み込み
		video = [Video(v_path,parameter_path,output_path) for v_path in video_dir_path]

		#施行毎の結果を保存するためのディレクトリを作成
		output_path_trial = output_path + "/trial{}".format(trial)
		if not os.path.exists(output_path_trial):
			print("ディレクトリ:trial{} を作成します".format(trial))
			os.makedirs(output_path_trial)

		Reconst = VideoReconstruct(video, parameter_path, output_path_trial)
		Reconst.readAnalysisAreaData()
		
		#動画像の書き出し準備
		save_video_separate_frame = 300 #300フレームごとに保存
		frame_count_min = Reconst.frame_count_min
		frameWrite = []

		for t in range(frame_count_min):
			Reconst.changeFrameReconst()

			frameWrite.append(Reconst.reconstruct(t)) #再構成後のフレームを追加

			#動画を途中で保存
			if len(frameWrite) % save_video_separate_frame == 0:
				Reconst.saveVideo(frameWrite, t, trial)
				frameWrite.clear()

			#進捗確認
			if t % 30 == 0:
				print("Reconstruct : {}/{}".format(t, frame_count_min))
		if len(frameWrite) != 0: #残りのフレームを保存
			Reconst.saveVideo(frameWrite, save_video_separate_frame, trial)
			frameWrite.clear()

		Reconst.writeData()
		del video, Reconst




def main():

	try:
		readParameter()

		start = time.time()

		#面積で動作量を求める
		# analysisArea()

		#動画像の再構成
		videoReconstruct()

		elapsed_time = time.time() - start
		print("実行時間 : {} ".format(elapsed_time) + "[sec]")

	except Exception as e:
		import traceback
		traceback.print_exc()
		print(e)
		line_notify(e)
	else:
		line_notify("実行完了")
	


if __name__ == '__main__':
	main()