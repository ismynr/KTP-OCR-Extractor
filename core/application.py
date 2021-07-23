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
import datetime

# Konfigurasi jalur
ROOT_PATH = os.getcwd()
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORD.csv')
CITIES_REC_PATH = os.path.join(ROOT_PATH, 'data/CITIES.csv')
DISTRICTS_REC_PATH = os.path.join(ROOT_PATH, 'data/DISTRICTS.csv')
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

# Lakukan OCR awal pada informasi di kartu ID
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

# Hapus informasi yang tidak valid di result_raw
def strip_op(result_raw):
	result_list = result_raw.split('\n')
	new_result_list = []
	for tmp_result in result_list:
		if tmp_result.strip(' '):
			new_result_list.append(tmp_result)
	return new_result_list


# KTP_ number
# NIK_NUMBER pengenalan pencocokan template
# Atur pengakuan dari kiri ke kanan
def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # Bungkus bentuk yang ditemukan dengan persegi panjang terkecil x,y,h,w
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
										key=lambda b: b[1][i], reverse=reverse))
	return cnts, boundingBoxes


def return_id_number(image, gray):
	# ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)#Pemrosesan dua nilai
	# Tentukan fungsi kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Kembalikan fungsi kernel persegi panjang
	# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)  # Pucuk topi
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	# ksize=-1 setara dengan menggunakan filter sobel 3*3, karena fungsi Sobel akan memiliki nilai negatif setelah turunan, dan akan ada nilai yang lebih besar dari 255, dan gambar aslinya adalah uint8,
	# Jadi gambar yang dibuat oleh Sobel tidak memiliki cukup bit dan akan terpotong. Jadi gunakan tipe data bertanda 16-bit, yaitu cv2.CV_16S
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")  # Sobel tidak dapat ditampilkan setelahnya, kembalikan ke format uint8 asli, jika tidak, gambar tidak dapat ditampilkan
	# print (np.array(gradX).shape)
	# Hubungkan nomor bersama-sama dengan menutup operasi (perluas dulu, lalu korosi)
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	# THRESH_OTSU akan secara otomatis menemukan ambang batas yang sesuai, yang cocok untuk puncak ganda, dan parameter ambang batas perlu diatur ke 0
	thresh = cv2.threshold(gradX, 0, 255,
						   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# Satu lagi operasi penutupan
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel) # Satu lagi operasi penutupan
	threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Temukan kontur
	cnts = threshCnts
	cur_img = image.copy()
	cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
	copy = image.copy()
	locs = []
	# Lintasi kontur
	for (i, c) in enumerate(cnts):
		# Hitung persegi panjang
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		# Pilih area yang tepat, sesuai dengan tugas yang sebenarnya, di sini pada dasarnya adalah sekelompok empat angka
		# if ar >10:
		# if (w > 40 ) and (h > 10 and h < 20):
		# Tetap dalam antrean
		if h > 10 and w > 100 and x < 300:
			img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
			locs.append((x, y, w, h, w * h))
	# Urutkan kontur yang cocok berdasarkan area dari terbesar ke terkecil
	locs = sorted(locs, key=itemgetter(1), reverse=False)

	# print(locs[1][0])
	nik = image[locs[1][1] - 10:locs[1][1] + locs[1][3] + 10, locs[1][0] - 10:locs[1][0] + locs[1][2] + 10]
	# cv_show('nik', nik)

	# print(nik)
	text = image[locs[2][1] - 10:locs[2][1] + locs[2][3] + 10, locs[2][0] - 10:locs[2][0] + locs[2][2] + 10]

	# Baca gambar template
	img = cv2.imread("images/module_nik.jpeg")
	# Skala abu-abu
	ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Gambar biner
	ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]
	# Hitung kontur
	# cv2.findContours()Parameter yang diterima oleh fungsi adalah citra biner, yaitu hitam putih (bukan skala abu-abu), cv2.RETR_EXTERNAL hanya mendeteksi kontur luar, cv2.CHAIN_APPROX_SIMPLE hanya mempertahankan koordinat titik akhir
	# Setiap elemen dalam daftar yang dikembalikan adalah garis besar dalam gambar
	refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)#上颜色
	# cv2.imshow('888', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	refCnts = sort_contours(refCnts, method="left-to-right")[0]  # Urutkan, dari kiri ke kanan, atas ke bawah
	digits = {}

	# Lintasi setiap kontur
	for (i, c) in enumerate(refCnts):
		# Hitung persegi panjang pembatas dan ubah ukurannya ke ukuran yang sesuai
		(x, y, w, h) = cv2.boundingRect(c)
		# if
		roi = ref[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# Setiap nomor sesuai dengan setiap template
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

	# Hitung kontur setiap grup
	digitCnts, hierarchy_nik = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	nik_r = nik.copy()

	cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

	# Cari lokasi NIK
	gX = locs[1][0]
	gY = locs[1][1]
	gW = locs[1][2]
	gH = locs[1][3]

	ctx = sort_contours(digitCnts, method="left-to-right")[0]

	locs_x = []
	# Lintasi kontur
	for (i, c) in enumerate(ctx):
		# Hitung persegi panjang
		(x, y, w, h) = cv2.boundingRect(c)

		# Pilih area yang tepat, sesuai dengan tugas yang sebenarnya, di sini pada dasarnya adalah sekelompok empat angka

		if h > 10 and w > 10:
			img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
			locs_x.append((x, y, w, h))

	# digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
	output = []
	groupOutput = []
	# Hitung setiap nilai di setiap grup
	for c in locs_x:
		# Temukan kontur nilai saat ini dan ubah ukuran ke ukuran yang sesuai
		(x, y, w, h) = c
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# cv_show('roi',roi)

		# Hitung skor pertandingan
		scores = []

		# Hitung setiap skor dalam template
		for (digit, digitROI) in digits.items():
			# Pencocokan template
			result = cv2.matchTemplate(roi, digitROI,
									   cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# Dapatkan nomor yang paling cocok

		groupOutput.append(str(np.argmax(scores)))

	# Seri
	cv2.rectangle(image, (gX - 5, gY - 5),
				  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	# Gambar, konten teks, koordinat, font, ukuran, warna, ketebalan font

	#mendapat jawabannya
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
			IMAGE_PATH = image

			raw_df = pd.read_csv(LINE_REC_PATH, header=None)
			# Baca kamus kota
			cities_df = pd.read_csv(CITIES_REC_PATH, header=None)
			# Baca kamus desa
			districts_df = pd.read_csv(DISTRICTS_REC_PATH, header=None)
			# Baca kamus agama
			religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
			# Baca kamus pernikahan
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
						loc2index[(i, j)] = arg_max  # Kesamaan tertinggi

			# PROSES EXTRAKSI KTP 
			# pengolahan data
			last_result_list = []
			useful_info = False
			for i, tmp_line in enumerate(result_list):
				tmp_list = []
				for j, tmp_word in enumerate(tmp_line.split(' ')):
					tmp_word = tmp_word.strip(':')
					if (i, j) in loc2index:
						useful_info = True
						if loc2index[(i, j)] == NEXT_LINE:  # Garis baru
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
				# Proses dua baris pertama
				if 'PROVINSI' in tmp_data or 'KABUPATEN' in tmp_data or 'KOTA' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[1:]):
						# Hitung kesamaan
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in cities_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# Dapatkan indeks kesamaan kata maksimum
						arg_max = np.argmax(tmp_sim_np)
						if tmp_sim_np[arg_max] >= 0.6:
							tmp_data[tmp_index + 1] = cities_df[0].values[arg_max]

				if 'KECAMATAN' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[1:]):
						# Hitung kesamaan
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in districts_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# Dapatkan indeks kesamaan kata maksimum
						arg_max = np.argmax(tmp_sim_np)
						if tmp_sim_np[arg_max] >= 0.6:
							tmp_data[tmp_index + 1] = districts_df[0].values[arg_max]

				# Proses jalur KTP
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

				# Proses garis agama
				if 'Agama' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[1:]):
						# Hitung kesamaan
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in religion_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# Dapatkan indeks kata yang paling mirip
						arg_max = np.argmax(tmp_sim_np)
						if tmp_sim_np[arg_max] >= 0.6:
							tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]

				# Berurusan dengan garis pernikahan
				if 'Status' in tmp_data or 'Perkawinan' in tmp_data:
					for tmp_index, tmp_word in enumerate(tmp_data[2:]):
						# Hitung kesamaan
						tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word, tmp_word_) for
										tmp_word_ in marriage_df[0].values]
						tmp_sim_np = np.asarray(tmp_sim_list)
						# Dapatkan indeks kata yang paling mirip
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
						fPROVINSI = alpha_with_remove_first_end_spaces(fPROVINSI)
						if fPROVINSI is None:
							raise ValueError('Provinsi pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fPROVINSI) is False:
							raise ValueError('Provinsi pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					if 'KABUPATEN' in string:
						fKABUPATEN_KOTA = string.replace('KABUPATEN ', '')
						fKABUPATEN_KOTA = alpha_with_remove_first_end_spaces(fKABUPATEN_KOTA)
						if fKABUPATEN_KOTA is None:
							raise ValueError('Kabupaten pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fKABUPATEN_KOTA) is False:
							raise ValueError('Kabupaten pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
					if 'KOTA' in string:
						fKABUPATEN_KOTA = string.replace('KOTA ', '')
						fKABUPATEN_KOTA = alpha_with_remove_first_end_spaces(fKABUPATEN_KOTA)
						if fKABUPATEN_KOTA is None:
							raise ValueError('Kota pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fKABUPATEN_KOTA) is False:
							raise ValueError('Kota pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
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
						tglLahirReplace = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", word)[0]
						fTANGGAL_LAHIR = datetime.datetime.strptime(tglLahirReplace, '%d-%m-%Y').strftime('%Y-%m-%d')
						word = word.replace(tglLahirReplace, '')
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
						if re.search(' +', word):
							fRT = word.split(' ')[0].strip()
							fRW = word.split(' ')[1].strip()
							if re.search('/', word):
								if re.search('/', word) is None:
									raise ValueError('RT dan RW pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
								fRT = word.split('/')[0].strip()
								fRW = word.split('/')[1].strip()

						else:
							if re.search('/', word):
								if re.search('/', word) is None:
									raise ValueError('RT dan RW pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
								fRT = word.split('/')[0].strip()
								fRW = word.split('/')[1].strip()

						# if re.search(' +', word) is None:
						# 	raise ValueError('RT dan RW spasi pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						# else:
						# 	fRT = word.split(' ')[0].strip()
						# 	fRW = word.split(' ')[1].strip()
						# 	if re.search('/', word):
						# 		if re.search('/', word) is None:
						# 			raise ValueError('RT dan RW pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						# 		fRT = word.split('/')[0].strip()
						# 		fRW = word.split('/')[1].strip()

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
						fKECAMATAN = alpha_with_remove_first_end_spaces(fKECAMATAN)
						if fKECAMATAN is None:
							raise ValueError('Kecamatan pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
						if special_match(fKECAMATAN) is False:
							raise ValueError('Kecamatan pada KTP kurang jelas, silahkan foto dan upload ulang KTP')
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
						'origin' : last_result_list,
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