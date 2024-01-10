originalFile = open("test.txt", "r", encoding='utf-8')  # 打开文件
trainFile = open("test2.data", "w", encoding='UTF-8')
trainFile.truncate(0)

nr_flag = 0
pa_flag = 0
entity = ""

for line in originalFile:
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
                    entity = word.split('/')[0] + nextWord.split('/')[0]

        # 非实体
        if tag != "nr" and tag != "ns" and tag != "nt":
            if entity == "":
                entity = word.split('/')[0]
            for i in range(len(entity)):
                trainFile.write(entity[i]+"\tO\n")
            entity = ""
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
            for i in range(len(entity)):
                trainFile.write(entity[i]+"\t")
                if i == 0:
                    trainFile.write("B-" + tag.upper() + "\n")
                else:
                    if tag == "nt" and i == len(entity) - 1:
                        trainFile.write("E-" + tag.upper() + "\n")
                    else:
                        trainFile.write("I\n")
            entity = ""     # 清空实体名

    trainFile.write("\n")

originalFile.close()
trainFile.close()
