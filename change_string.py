def replace_in_file(file_path, old_str, new_str):
    # 파일 읽어들이기
    fr = open(file_path, 'r', encoding="utf-8")
    lines = fr.readlines()
    fr.close()
    
    # old_str -> new_str 치환
    fw = open(file_path, 'w', encoding="utf-8")
    for line in lines:
        fw.write(line.replace(old_str, new_str))
    fw.close()

# 호출: file1.txt 파일에서 comma(,) 없애기
replace_in_file("test.json",'{},\n', '')
