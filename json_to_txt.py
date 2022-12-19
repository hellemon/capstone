# -*- coding: UTF-8 -*- 

import os
import io
import re
import json

# 폴더 위치
dir_root = r'C:\186.복지 분야 콜센터 상담데이터\01.데이터'
dir_data = ['1.Training', '2.Validation']
dir_type = ['라벨링데이터', '원천데이터']
# HOS : 01, MOB : 02, MEN : 03
# Number Start 1 ~ n
dir_path = [['TL1_01.대학병원', 'TL2_02.광역이동지원센터', 'TL3_03.정신건강복지센터'],['VL1_01.대학병원', 'VL2_02.광역이동지원센터', 'VL3_03.정신건강복지센터']]
dir_path2 = ['01.대학병원', '02.광역이동지원센터', '03.정신건강복지센터']
dir_path3 = [['01.진료안내', '02.병원이용안내', '03.민원'],
['01.상담', '02.고객대응', '03.민원'], ['01.정신건강상담', '02.자살위기개입']]
dir_depth = [
	[['01.검사','02.입원','03.외래','04.응급','05.건강검진'], ['01.시설안내','02.입퇴원','03.증명서발급','04.원무상담','05.장례식장안내'], ['01.외래진료불만','02.검사불만','03.치료불만','04.응급실불만','05.기타서비스불만']], # HOS : 01
	[['01.적용기준','02.규정문의'], ['01.차량요청','02.예약변경 및 취소'], ['01.예약불만','02.기사관련불만','03.규정불만','04.이용제한','05.서비스개선요청']], # MOB : 02
	[['01.조현병','02.우울증','03.조울증','04.불안장애','05.물질중독','06.행위중독','07.치매','08.기타'], ['01.가정불화','02.경제문제','03.이성문제','04.신체정신적문제','05.직장문제','06.외로움고독','07.학교성적진로','08.친구동료문제','09.기타']] # MEN : 03
	]

log_file = io.open('00_json_to_txt.log', 'w', encoding='UTF-8')

def clean_text(inputString):

	text_rmv = re.sub('[-=+,#/\?:^.@*\'※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', inputString)
	text_rmv = ' '.join(text_rmv.split())
	return text_rmv

def writeTXTFile(fname, wTxt):
	
	writeTxt = clean_text(wTxt)
	print('CLEAN TEXT = ' + writeTxt)
	txt_file = io.open(fname, 'w', encoding='UTF-8')
	try : 
		if len(writeTxt) > 0:
			txt_file.write(writeTxt)
		else:
			print('txt BLANK Error : ' + fname)
			s_log = 'blank TXT Error : not in contents text \n' + fname + ' : ' + wTxt + '\n'
			log_file.write(s_log)
	except Exception as e:
		print('txt WRITE Error : ' + str(e))
		s_log = 'write TXT Error : ' + str(e) + '\n' + fname + ' : ' + wTxt + '\n'
		log_file.write(s_log)
	txt_file.close()

def readJSONFile(ffile):

	try : 
		with open(ffile, encoding='utf-8') as json_file:
			json_data = json.load(json_file)
			json_arr = json_data.get('inputText')
			for jlist in json_arr:
				orgText = jlist.get('orgtext')
				print('JSON TEXT = ' + orgText)
				tfile = ffile.replace('.json', '.txt')
				writeTXTFile(tfile, orgText)
				break
	except Exception as e:
		print('read JSON Error : ' + str(e))
		s_log = 'read JSON Error : ' + (str(e)) + '\n' + ffile + '\n'
		log_file.write(s_log)

def searchFile(dirname):
	if False == os.path.isdir(dirname):
		pass
	else:
		filenames = os.listdir(dirname)
		for filename in filenames:
			print('source = ' + filename)
			full_filename = os.path.join(dirname, filename)
			if os.path.isdir(full_filename):
				searchFile(full_filename)
			else:
				path = os.path.dirname(full_filename)
				ext = os.path.splitext(full_filename)[-1]
				if ext == '.json' :
					print('target = ' + full_filename)
					readJSONFile(full_filename)

def setTargetFolder():

	# searchFile('/run/media/demo/KLCUBE/data/03_valid/TXT/')
	
	dir_target = dir_root
	for d in dir_data:
		dir_target = dir_root
		d_target = dir_target + '\\' + d	# /dir_root/01_train
		for dt in dir_type:
			if dt == '라벨링데이터':
				dt_target = d_target + '\\' + dt	# /dir_root/01_train/TXT
				iscount = 0
				if d == '1.Training':
					nextpath = dir_path[0]
				else:
					nextpath = dir_path[1]
				for dp in nextpath:
					iscount += 1
					dp_target = dt_target + '\\' + dp	# /dir_root/01_train/TXT/01
					dp2_count = 0
					for dp2 in dir_path2:
						dp2_count += 1
						dp3_count = 0
						if dp2_count != iscount:
							continue
						dd_target = dp_target + '\\' + dp2	# /dir_root/01_train/TXT/01/01
						for dp3 in dir_path3:
							for ddp in dp3:
								dp3_target = dd_target + '\\' + ddp # /dir_root/01_train/TXT/01/01/01

								for dd in dir_depth:												
										for k in dd:
											for dtgt in k:
												dtgt_target = dp3_target + '\\' + dtgt # /dir_root/01_train/TXT/01/01/01
												print('target folder = ' + dtgt_target)
												searchFile(dtgt_target)
								
						
	

setTargetFolder()
log_file.close()
print('00_json_to_txt.py : End work')

