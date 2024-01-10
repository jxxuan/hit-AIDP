f = open("D:\AIPrac\text\199801.txt", "r", encoding='ANSI')  # 打开文件
# 变量
dic = {}    # 用字典dic统计各实体出现次数
nr_count = 0    # 各类型计数
ns_count = 0
nt_count = 0
nx_count = 0
nz_count = 0
nr_flag = 0     # 人名标志，用于判断合并中文姓名
pa_flag = 0     # 括号标志，用于判断实体是否在括号中
entity = ""     # 存储实体名

for line in f:
    if len(line) != 0:  # 行非空
        words = line.split()[1:]    # 以空格分词为列表words并去掉句子标识
    else:   # 跳过空行
        continue

    for word in words:
        tag = word.split('/')[1]    # 获得词的类型

        if nr_flag == 1:    # 连续两个nr，在处理前一个词时被合并为一个中文名，跳过此轮循环
            nr_flag = 0
            continue

        if word[0] == "[":  # 括号开始
            pa_flag = 1
            entity = word.split('/')[0][1:]     # 获取嵌套实体的第一个实体名并去掉括号“[”
            continue
        if pa_flag == 1:    # 词在括号中，将其拼接进实体名
            entity = entity + word.split('/')[0]
            if "]" in word:     # 括号结束
                tag = word.split(']')[1]
            else:
                continue

        if tag == "nr":     # 处理人名，若后一个也是人名则拼接
            if words.index(word) != len(words) - 1:
                nextWord = words[words.index(word) + 1]
                if nextWord.split('/')[1] == "nr":
                    nr_flag = 1
                    nr_count += 1
                    entity = word.split('/')[0] + nextWord.split('/')[0]
            else:
                nr_count += 1
        # 计数
        if tag == "ns":
            ns_count += 1
        if tag == "nt":
            nt_count += 1
        if tag == "nx":
            nx_count += 1
        if tag == "nz":
            nz_count += 1
        # 非实体
        if tag != "nr" and tag != "ns" and tag != "nt" and tag != "nx" and tag != "nz":
            continue
        # 处理人名和括号的特殊情况：实体名已拼接完毕
        if pa_flag == 1:
            pa_flag = 0
        else:   # 非特殊情况，直接获取实体名
            if not nr_flag == 1:
                entity = word.split('/')[0]
        # 排除实体名为空
        if len(entity) == 0:
            continue
        else:
            if entity in dic:
                dic[entity] += 1
            else:
                dic[entity] = 1
            entity = ""     # 清空实体名
# 将字典转为二元组列表进行排序
dic_sorted = sorted(list(zip(dic.values(), dic.keys())), reverse=True)
# 结果输出
print("各实体类型出现数量：")
print("nr:%d, ns:%d, nt:%d, nx:%d, nz:%d" % (nr_count, ns_count, nt_count, nx_count, nz_count))
print("出现次数最多的前十个实体及其次数：")
for i in range(9):
    print(list(dic_sorted)[i][1] + "，%d次；" % list(dic_sorted)[i][0], end='')
print(list(dic_sorted)[9][1] + "，%d次。" % list(dic_sorted)[9][0])
