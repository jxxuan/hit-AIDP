from gensim import corpora
from collections import defaultdict

# 将字符串切分为单词，并去掉停用词
stoplist = set('for a of the and to in'.split(' '))
text = 'This is a sample sentence, showing off the stop words filtration.'
words = [word for word in text.lower().split() if word not in stoplist]

# 统计每个词的出现频度，只出现1次的词，作为噪音去掉
frequency = defaultdict(int)
for word in words:
    frequency[word] += 1
processed_text = [word for word in words if frequency[word] > 1]

# 使用gensim.corpora.Dictionary库将字符串转化为id的列表
dictionary = corpora.Dictionary([processed_text])
corpus = [dictionary.doc2bow([text]) for text in processed_text]
