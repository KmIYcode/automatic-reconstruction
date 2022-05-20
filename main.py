from analysis_area import AnalysisArea
from video_reconst import VideoReconstruct
from video import Video
import glob
import time
import os

source_path = "./data/"
parameter_path = source_path + "/param" #./sourceフォルダ内のparemeterフォルダへのパス
output_path = "./out"
if not os.path.exists(output_path):
	os.makedirs(self.output_path)

print("source_path : {}".format(source_path))

DIFF = 3 #差分フレーム


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

	#動画を読み込み
	video = [Video(v_path,parameter_path,output_path) for v_path in video_dir_path]

	Reconst = VideoReconstruct(video, parameter_path, output_path)
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
			Reconst.saveVideo(frameWrite, t)
			frameWrite.clear()

		#進捗確認
		if t % 30 == 0:
			print("Reconstruct : {}/{}".format(t, frame_count_min))
	if len(frameWrite) != 0: #残りのフレームを保存
		Reconst.saveVideo(frameWrite, save_video_separate_frame)
		frameWrite.clear()

	Reconst.writeData()
	del video, Reconst




def main():


	start = time.time()

	#面積で動作量を求める
	analysisArea()

	#動画像の再構成
	videoReconstruct()

	elapsed_time = time.time() - start
	print("実行時間 : {} ".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
	main()