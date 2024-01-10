import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
import logging
import string

# 装载英文Bert tokenizer和bert-base-cased模型
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

# 用Bert为句子"Apple company does not sell the apple."编码
sentence = "Apple company does not sell the apple."
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)
# 输出句子转化后的ID
sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Sentence IDs:", sentence_ids)
print("Decoded tokens:", tokenizer.convert_ids_to_tokens(sentence_ids))

# 分别输出句子编码后单词'CLS'、'Apple'、'apple'和'SEP'四个词对应的编码
print("Word IDs:")
for word in ['[CLS]', 'Apple', 'apple', '[SEP]']:
    print(word, tokenizer.encode(word, add_special_tokens=False)[0])

# 分别计算'Apple'和'apples'、'[CLS]'和'Apple'、'[CLS]'和'[SEP]'之间的距离
words_list = [('Apple', 'apples'), ('[CLS]', 'Apple'), ('[CLS]', '[SEP]')]
embedding = model.get_input_embeddings().weight
for words in words_list:
    word0vec = embedding[tokenizer.encode(words[0], add_special_tokens=False)[0]]
    word1vec = embedding[tokenizer.encode(words[1], add_special_tokens=False)[0]]
    distant = torch.dist(word0vec, word1vec).item()
    print(f"Distance between {words[0]} and {words[1]}:{distant}")

# 输入句子"I have a [MASK] named Charlie."，重新加载BertForMaskedLM模型，通过bert预测[mask]位置最可能的单词
logging.getLogger("transformers").setLevel(logging.ERROR)
masked_sentence = "I have a [MASK] named Charlie."
masked_token_ids = tokenizer.encode(masked_sentence, add_special_tokens=True, padding='max_length', truncation=True, max_length=10, return_tensors='pt')
# 创建 attention_mask
attention_mask = torch.ones_like(masked_token_ids)
bert_masked_model = BertForMaskedLM.from_pretrained('bert-base-cased')
outputs = bert_masked_model(input_ids=masked_token_ids, attention_mask=attention_mask)
predictions = torch.argmax(outputs.logits, dim=-1)
predicted_word = tokenizer.decode(predictions[0][masked_token_ids.squeeze(0) == tokenizer.mask_token_id].item())
predicted_word = predicted_word.replace(" ", "")
print("Predicted word:", predicted_word)

# 输入句子“I have a cat.”，重新加载BertForNextSentencePrediction模型，通过bert 预测下一句。
this_sentence = "I have a cat."
masked_sentence = this_sentence
max_length = 10
predicted_sentence = ''
for i in range(max_length):
    masked_sentence = masked_sentence + " [MASK]"
    masked_token_ids = tokenizer.encode(masked_sentence, add_special_tokens=True, padding='max_length', truncation=False,
                                        max_length=2 * max_length, return_tensors='pt')
    attention_mask = torch.ones_like(masked_token_ids)
    outputs = bert_masked_model(input_ids=masked_token_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_word = tokenizer.decode(predictions[0][masked_token_ids.squeeze(0) == tokenizer.mask_token_id].item())
    predicted_word = predicted_word.replace(" ", "")
    masked_sentence = masked_sentence.replace("[MASK]", predicted_word)
    predicted_sentence = predicted_sentence + predicted_word + " "
    if predicted_word in string.punctuation:
        break
print("Predicted sentence:", predicted_sentence)
