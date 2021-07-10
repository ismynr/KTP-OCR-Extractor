# coding=utf-8
import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests
from flask import request
import re

import os
import textdistance #belum ada di requirenments
import pandas as pd
from operator import itemgetter, attrgetter
import json

# 路径配置
ROOT_PATH = os.getcwd()
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORD.csv')
CITIES_REC_PATH = os.path.join(ROOT_PATH, 'data/CITIES.csv')
RELIGION_REC_PATH = os.path.join(ROOT_PATH, 'data/RELIGION.csv')
MARRIAGE_REC_PATH = os.path.join(ROOT_PATH, 'data/MARRIAGE.csv')
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3

def special_match(strg, search=re.compile(r'/^[a-zA-Z]+$/').search):
	return not bool(search(strg))

def numeric_match(strg, search=re.compile(r'/[^0-9]/').search):
	return not bool(search(strg))

def alpha_with_remove_first_end_spaces(string):
	return re.compile('[^a-zA-Z\s]').sub('', string).rstrip().lstrip();


def allowed_image(filename):
	allow = ["JPEG", "JPG", "PNG"]

	if not "." in filename:
		return False

	ext = filename.rsplit(".", 1)[1]

	if ext.upper() in allow:
		return True
	else:
		return False

# 对身份证上的信息进行初步OCR
def ocr_raw(image_path):
	# (1) Read
	img_raw = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_UNCHANGED)
	# img_raw = cv2.imread(image_path)
	image = cv2.resize(img_raw, (50 * 16, 500))
	img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	id_number = return_id_number(image, img_gray)
	# img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
	# (2) Threshold
	cv2.fillPoly(img_gray, pts=[np.asarray([(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))
	th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
	# (3) Detect
	result_raw = pytesseract.image_to_string(threshed, lang="ind")
	return result_raw, id_number

# 去除result_raw中的无效信息
def strip_op(result_raw):
	result_list = result_raw.split('\n')
	new_result_list = []
	for tmp_result in result_list:
		if tmp_result.strip(' '):
			new_result_list.append(tmp_result)
	return new_result_list


# KTP_ number
# NIK_NUMBER模版匹配识别
# 从左到右进行排列识别
def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
										key=lambda b: b[1][i], reverse=reverse))
	return cnts, boundingBoxes


def return_id_number(image, gray):
	# ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)#二值处理
	# 定义核函数
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 返回矩形核函数
	# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)  # 顶帽
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	# ksize=-1相当于用3*3的 sobel过滤器，由于 Sobel 函数求导数后会有负值，还会有大于 255 的值，而原图像是 uint8 ，
	# 所以 Sobel 建立的图像位数不够，会有截断。因此要使用 16 位有符号的数据类型，即 cv2.CV_16S
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")  # Sobel之后无法显示，将其转回原来的 uint8 形式，否则无法显示图像
	# print (np.array(gradX).shape)
	# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
	thresh = cv2.threshold(gradX, 0, 255,
						   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# 再来一个闭操作
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)  # 再来一个闭操作
	threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
	cnts = threshCnts
	cur_img = image.copy()
	cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
	copy = image.copy()
	locs = []
	# 遍历轮廓
	for (i, c) in enumerate(cnts):
		# 计算矩形
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
		# if ar >10:
		# if (w > 40 ) and (h > 10 and h < 20):
		# 符合的留下来
		if h > 10 and w > 100 and x < 300:
			img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
			locs.append((x, y, w, h, w * h))
	# 将符合的轮廓按照面积从大到小排序
	locs = sorted(locs, key=itemgetter(1), reverse=False)

	# print(locs[1][0])
	nik = image[locs[1][1] - 10:locs[1][1] + locs[1][3] + 10, locs[1][0] - 10:locs[1][0] + locs[1][2] + 10]
	# cv_show('nik', nik)

	# print(nik)
	text = image[locs[2][1] - 10:locs[2][1] + locs[2][3] + 10, locs[2][0] - 10:locs[2][0] + locs[2][2] + 10]

	# 读取一个模板图像
	img = cv2.imread("images/module.jpeg")
	# 灰度图
	ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 二值图像
	ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]
	# 计算轮廓
	# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
	# 返回的list中每个元素都是图像中的一个轮廓
	refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)#上颜色
	# cv2.imshow('888', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	refCnts = sort_contours(refCnts, method="left-to-right")[0]  # 排序，从左到右，从上到下
	digits = {}

	# 遍历每一个轮廓
	for (i, c) in enumerate(refCnts):
		# 计算外接矩形并且resize成合适大小
		(x, y, w, h) = cv2.boundingRect(c)
		# if
		roi = ref[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# 每一个数字对应每一个模板
		digits[i] = roi
	# cv_show('digits[i]', digits[i])
	# nik = np.clip(nik, 0, 255)
	# nik = np.array(nik,np.uint8)
	# NIK识别
	gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
	# gray_nik = cv2.GaussianBlur(gray_nik, (3, 3), 0)
	# ret_nik, thresh_nik = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY)
	# cv2.imshow('8', gray_nik)
	# cv2.waitKey(0)
	group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]
	# cv2.imshow('9', group)

	# 计算每一组的轮廓
	digitCnts, hierarchy_nik = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	nik_r = nik.copy()

	cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

	# 定位NIK的位置
	gX = locs[1][0]
	gY = locs[1][1]
	gW = locs[1][2]
	gH = locs[1][3]

	ctx = sort_contours(digitCnts, method="left-to-right")[0]

	locs_x = []
	# 遍历轮廓
	for (i, c) in enumerate(ctx):
		# 计算矩形
		(x, y, w, h) = cv2.boundingRect(c)

		# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组

		if h > 10 and w > 10:
			img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
			locs_x.append((x, y, w, h))

	# digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
	output = []
	groupOutput = []
	# 计算每一组中的每一个数值
	for c in locs_x:
		# 找到当前数值的轮廓，resize成合适的的大小
		(x, y, w, h) = c
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# cv_show('roi',roi)

		# 计算匹配得分
		scores = []

		# 在模板中计算每一个得分
		for (digit, digitROI) in digits.items():
			# 模板匹配
			result = cv2.matchTemplate(roi, digitROI,
									   cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# 得到最合适的数字

		groupOutput.append(str(np.argmax(scores)))

	# 画出来
	cv2.rectangle(image, (gX - 5, gY - 5),
				  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	# 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度

	# 得到结果
	output.extend(groupOutput)
	return ''.join(output)
	
def extract_ktp():
	# KTP ENTITY
	fPROVINSI = ""
	fKABUPATEN_KOTA = ""
	fNIK = ""
	fNAMA = ""
	fTEMPAT = ""
	fTANGGAL_LAHIR = ""
	fJENIS_KELAMIN = ""
	fGOL_DARAH = ""
	fALAMAT = ""
	fRT = ""
	fRW = ""
	fKEL_DESA = ""
	fKECAMATAN = ""
	fAGAMA = ""
	fSTATUS_PERKAWINAN = ""
	fPEKERJAAN = ""
	fKEWARGANEGARAAN = ""
	fBERLAKU_HINGGA = ""

	if request.method == "POST":
		image = request.files["ktp"]

		if image.filename == "":
			return {
				'success':False,
				'message':'Empty Fields!'
			}

		if allowed_image(image.filename):
			# img = cv2.imdecode(np.fromstring(request.files['ktp'].read(), np.uint8), cv2.IMREAD_UNCHANGED)  
			# return {
			# 	'msg': image.filename
			# }
			# img = cv2.imdecode(np.fromstring(request.files['ktp'].read(), np.uint8), cv2.IMREAD_UNCHANGED)  
			# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
			# result = pytesseract.image_to_string((threshed), lang="ind")
			# result.replace('\n', ' ')

			IMAGE_PATH = image

			raw_df = pd.read_csv(LINE_REC_PATH, header=None)
			# 读入城市字典
			cities_df = pd.read_csv(CITIES_REC_PATH, header=None)
			# 读入宗教字典
			religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
			# 读入婚姻字典
			marriage_df = pd.read_csv(MARRIAGE_REC_PATH, header=None)
			result_raw, id_number = ocr_raw(IMAGE_PATH)
			result_list = strip_op(result_raw)

			# result_list = ['—————— ————— —&——— ——————', 'PROVINSI DAEGAH ISTIMAWA YOGYEKARTA', 'KABUPATEN SLEMQN', 'NIK : 347LL40209790001', 'Nama :RIYANTO. SE', 'Tempat/Tgi Lahir : GROBOGAN. 02-09-1979', 'Jenis Kelamin : LAKI-LAKI Gol Darah :O', 'Alamat PRM PURI DOMAS D-3. SEMPU', 'RTARW 001: 024', 'Kel/Desa : WEDOMARTANI', 'Kecamatan : NGEMPLAK', 'Agama :ISPAM', 'Status Perkawman: KAWAN PE', 'Pekerjaan : PEDAGANG 05-06-2012', 'Kewarganegaraan: WNI SA—', 'Berlaku Hingga  :02-09-2017 N EA']
			loc2index = dict()
			for i, tmp_line in enumerate(result_list):
				for j, tmp_word in enumerate(tmp_line.split(' ')):
					tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_, tmp_word.strip(':')) for
									tmp_word_ in
									raw_df[0].values]
					tmp_sim_np = np.asarray(tmp_sim_list)
					arg_max = np.argmax(tmp_sim_np)
					if tmp_sim_np[arg_max] >= 0.6:
						loc2index[(i, j)] = arg_max  ##最高相似度

			# PROSES EXTRAKSI KTP 
			# 数据处理
			last_result_list = []
			useful_info = False
			for i, tmp_line in enumerate(result_list):
				tmp_list = []
				for j, tmp_word in enumerate(tmp_line.split(' ')):
					tmp_word = tmp_word.strip(':')
					if (i, j) in loc2index:
						useful_info = True
						if loc2index[(i, j)] == NEXT_LINE:  ##换行
							last_result_list.append(tmp_list)
							tmp_list = []
						tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
						if loc2index[(i, j)] in NEED_COLON:
							tmp_list.append(':')
					elif tmp_word == ':' or tmp_word == '':
						continue
					else:
						tmp_list.append(tmp_word)
				if useful_info:
					if len(last_result_list) > 2 and ':' not in tmp_list:
						last_result_list[-1].extend(tmp_list)
					else:
						last_result_list.append(tmp_list)
			# print(last_result_list)

			for tmp_data in last_result_list:
				if '—' in tmp_data:
					tmp_data.remove('—')
				# 对前两行进行处理
				if 'PROVINSI' in tmp_data or 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[1:]):
						# 计算相似度
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in cities_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# 获得相似度最大单词索引
						arg_max = np.argmax(tmp_sim_np)
						if tmp_sim_np[arg_max] >= 0.6:
							tmp_data[tmp_index + 1] = cities_df[0].values[arg_max]

				# 对身份证一行进行处理
				if 'NIK' in tmp_data:
					if len(id_number) != 16:
						id_number = tmp_data[2]
						if "D" in id_number:
							id_number = id_number.replace("D", "0")
						if "?" in id_number:
							id_number = id_number.replace("?", "7")
						if "L" in id_number:
							id_number = id_number.replace("L", "1")
						while len(tmp_data) > 2:
							tmp_data.pop()
						tmp_data.append(id_number)
					else:
						while len(tmp_data) > 3:
							tmp_data.pop()
						tmp_data[2] = id_number

				# 对宗教一行进行处理
				if 'Agama' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[1:]):
						# 计算相似度
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in religion_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# 获得相似度最大单词索引
						arg_max = np.argmax(tmp_sim_np)
						if tmp_sim_np[arg_max] >= 0.6:
							tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]

				# 对婚姻一行进行处理
				if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[2:]):
						# 计算相似度
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in marriage_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# 获得相似度最大单词索引
						arg_max = np.argmax(tmp_sim_np)
						if tmp_sim_np[arg_max] >= 0.6:
							tmp_data[tmp_index + 2] = marriage_df[0].values[arg_max]
				if 'Alamat' in tmp_data:
					for tmp_index in range(len(tmp_data)):
						if "!" in tmp_data[tmp_index]:
							tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
						if "1" in tmp_data[tmp_index]:
							tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
						if "i" in tmp_data[tmp_index]:
							tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
			
			result_text = ""
			for tmp_data in last_result_list:
			 	result_text += ' '.join(tmp_data)+' '

			# yg dipake NIK, AGAMA, KEL_DESA, NAMA, PEKERJAAN, RT, RW, TANGGAL_LAHIR, TEMPAT
			# ngecek atribut yang dipakai --------- START
			if 'NIK' in result_text and 'Nama' in result_text and 'Tempat/Tgl Lahir' in result_text and 'Jenis Kelamin' in result_text and 'RT/RW' in result_text and 'Kel/Desa' in result_text and 'Agama' in result_text and 'Pekerjaan' in result_text:
				pass
			else:
				return {
					'success':False,
					'origin' : last_result_list,
					'message':'Atributte KTP ada yang tidak terdeteksi, mohon upload ulang !'
				}

			# ngecek atribut yang dipakai --------- END

			# mengambil atribut --------- START
			for tmp_data in last_result_list:
				# ngecek kalo ada exception --------- START
				try:
					string = ' '.join(tmp_data)
					if 'PROVINSI' in string:
						fPROVINSI = string.replace('PROVINSI ', '')
					if 'KABUPATEN' in string:
						fKABUPATEN_KOTA = string.replace('KABUPATEN ', '')
					if 'KOTA' in string:
						fKABUPATEN_KOTA = string.replace('KOTA ', '')
					# PENTING DIPAKE
					if 'NIK' in string:
						fNIK = string.replace('NIK : ', '')
						if fNIK is None:
							raise ValueError('NIK pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if numeric_match(fNIK) is False:
							raise ValueError('NIK pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if len(fNIK) < 16:
							raise ValueError('NIK pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					# PENTING DIPAKE
					if 'Nama' in string:
						fNAMA = string.replace('Nama : ', '')
						fNAMA = alpha_with_remove_first_end_spaces(fNAMA)
						if fNAMA is None or special_match(fNAMA) is False:
							raise ValueError('Nama pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					# PENTING DIPAKE
					if 'Tempat/Tgl Lahir' in string:
						word = string.replace('Tempat/Tgl Lahir : ', '')
						if re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", word) is None:
							raise ValueError('Tempat Lahir pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						fTANGGAL_LAHIR = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", word)[0]
						word = word.replace(fTANGGAL_LAHIR, '')
						fTEMPAT = alpha_with_remove_first_end_spaces(word)
						if fTEMPAT is None:
							raise ValueError('Tempat Lahir pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fTEMPAT) is False:
							raise ValueError('Tempat Lahir pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					if 'Jenis Kelamin' in string:
						fJENIS_KELAMIN = string.replace('Jenis Kelamin : ', '')
					if 'Gol. Darah' in string:
						fGOL_DARAH = string.replace('Gol. Darah : ', '')
					if 'Alamat' in string:
						fALAMAT = string.replace('Alamat : ', '')
					# PENTING DIPAKE
					if 'RT/RW' in string:
						word = string.replace("RT/RW : ",'')
						if re.search('/', word) is None:
							raise ValueError('RT dan RW pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						fRT = word.split('/')[0].strip()
						fRW = word.split('/')[1].strip()
						if fRT is None:
							raise ValueError('RT pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if numeric_match(fRT) is False:
							raise ValueError('RT pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if fRW is None:
							raise ValueError('RW pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if numeric_match(fRW) is False:
							raise ValueError('RW pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					# PENTING DIPAKE
					if 'Kel/Desa' in string:
						fKEL_DESA = string.replace('Kel/Desa : ', '')
						fKEL_DESA = alpha_with_remove_first_end_spaces(fKEL_DESA)
						if fKEL_DESA is None:
							raise ValueError('Kel Desa pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fKEL_DESA) is False:
							raise ValueError('Kel Desa pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					if 'Kecamatan' in string:
						fKECAMATAN = string.replace('Kecamatan : ', '')
					# PENTING DIPAKE
					if 'Agama' in string:
						fAGAMA = string.replace('Agama : ', '')
						fAGAMA = alpha_with_remove_first_end_spaces(fAGAMA)
						if fAGAMA is None:
							raise ValueError('Agama pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fAGAMA) is False:
							raise ValueError('Agama pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					if 'Status Perkawinan' in string:
						fSTATUS_PERKAWINAN = string.replace('Status Perkawinan : ', '')
					# PENTING DIPAKE
					if 'Pekerjaan' in string:
						fPEKERJAAN = string.replace('Pekerjaan : ', '')
						fPEKERJAAN = alpha_with_remove_first_end_spaces(fPEKERJAAN)
						if fPEKERJAAN is None:
							raise ValueError('Pekerjaan pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fPEKERJAAN) is False:
							raise ValueError('Pekerjaan pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					if 'Kewarganegaraan' in string:
						fKEWARGANEGARAAN = string.replace('Kewarganegaraan : ', '')
					if 'Berlaku Hingga' in string:
						fBERLAKU_HINGGA = string.replace('Berlaku Hingga : ', '')
				except ValueError as e:
					return {
						'success':False,
						'message': str(e)
					}
				except Exception:
					return {
						'success':False,
						'origin' : last_result_list,
						'message':'KTP kurang jelas, mohon upload ulang !'
					}
				# ngecek kalo ada exception --------- END
				
			return {
				'success':True,
				'message':'Success!',
				'origin' : last_result_list,
				'data': {
					'PROVINSI': fPROVINSI,
					'KABUPATEN_KOTA': fKABUPATEN_KOTA,
					'NIK': fNIK,
					'NAMA': fNAMA,
					'TEMPAT': fTEMPAT,
					'TANGGAL_LAHIR': fTANGGAL_LAHIR,
					'JENIS_KELAMIN': fJENIS_KELAMIN,
					'GOL_DARAH': fGOL_DARAH,
					'ALAMAT': fALAMAT,
					'RT': fRT,
					'RW': fRW,
					'KEL_DESA': fKEL_DESA,
					'KECAMATAN': fKECAMATAN,
					'AGAMA': fAGAMA,
					'STATUS_PERKAWINAN': fSTATUS_PERKAWINAN,
					'PEKERJAAN': fPEKERJAAN,
					'KEWARGANEGARAAN': fKEWARGANEGARAAN,
					'BERLAKU_HINGGA': fBERLAKU_HINGGA,
				}
			}
			# mengambil atribut --------- END
			
		else:   
			return {
				'success':False,
				'message':'Extensi File Tidak Diijinkan, file KTP harus berupa jpg, jpeg, png'
			}
	else:
		return {
			'success':False,
			'message':'Request Method Tidak Diijinkan!'
		}