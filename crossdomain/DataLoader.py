# coding=utf-8
import codecs
import re
from copy import deepcopy
import numpy as np
import math

PAD = "<PAD>"
UNK = "<UNK>"


def zero_digits(s):
	"""
	Replace every digit in a string by a zero.
	"""
	return re.sub('\d', '0', s)



def get_vocab(filename):
	vocab = {}
	i = 0
	for line in codecs.open(filename, encoding='utf-8'):
		line = line.strip()
		vocab[line] = i
		i = i + 1
	return vocab

def get_nums(datas, vocab, mode='word'):
	nums = []
	for data in datas:
		if mode == "word":
			num = [vocab[d if d in vocab else UNK] for d in data]
		else:
			num = [vocab[d] for d in data]
		nums.append(num)
	return nums


def train_batch(datasets, word_vocab, tag_vocab, char_vocab):
	while True:
		for sen_batch, char_batch, tag_batch, len_batch, sen_max_lens in datasets:
			sen_num_batch = get_nums(sen_batch, word_vocab)
			chars_num_batch = get_nums(char_batch, char_vocab, 'char')
			tags_num_batch = get_nums(tag_batch, tag_vocab, 'tag')
			yield sen_num_batch, chars_num_batch, tags_num_batch, len_batch, sen_max_lens


def test_batch(datasets, word_vocab, tag_vocab, char_vocab):
	for sen_batch, char_batch, tag_batch, len_batch, sen_max_lens in datasets:
		sen_num_batch = get_nums(sen_batch, word_vocab)
		chars_num_batch = get_nums(char_batch, char_vocab, 'char')
		tags_num_batch = get_nums(tag_batch, tag_vocab, 'tag')
		yield sen_num_batch, chars_num_batch, tags_num_batch, len_batch, sen_max_lens



def dsr_train_batch(datasets, word_vocab, tag_vocab, char_vocab, domain):
	while True:
		for sen_batch, char_batch, tag_batch, len_batch, sen_max_lens in datasets:
			sen_num_batch = get_nums(sen_batch, word_vocab)
			chars_num_batch = get_nums(char_batch, char_vocab, 'char')
			tags_num_batch = get_nums(tag_batch, tag_vocab, 'tag')
			domain_batch = generate_domain_batch(domain, len(sen_batch))
			yield sen_num_batch, chars_num_batch, tags_num_batch, len_batch, sen_max_lens, domain_batch


def dsr_test_batch(datasets, word_vocab, tag_vocab, char_vocab):
	for sen_batch, char_batch, tag_batch, len_batch, sen_max_lens in datasets:
		sen_num_batch = get_nums(sen_batch, word_vocab)
		chars_num_batch = get_nums(char_batch, char_vocab, 'char')
		tags_num_batch = get_nums(tag_batch, tag_vocab, 'tag')
		yield sen_num_batch, chars_num_batch, tags_num_batch, len_batch, sen_max_lens


def generate_domain_batch(domain, batch_len):
	return [0] * batch_len if domain == "source" else [1]*batch_len

class Datasets(object):

	def __init__(self, file, batch_size, mode, word_max_length, sort):
		self.file = file
		self.length = None
		self.batch_size = batch_size
		self.mode = mode
		self.word_max_length = word_max_length
		self.datas = self.load_all_data()
		self.sort = sort
		self.sort_datas = sorted(self.datas, key=lambda data: data[1])
		print("all_count is：" + str(len(self.datas)))
		print("total batch is: " + str(math.ceil(float(len(self.datas)) / self.batch_size)))

	def __len__(self):

		if self.length is None:
			self.length = len(self.datas)
		return self.length

	def __iter__(self):
		# 全部的batch数据(已经是batch了)
		all_batch = self.get_all_batch()
		for batch in all_batch:
			no_lower_sen_batch, sen_batch, char_batch, tag_batch, len_batch, sen_max_lens, = [], [], [], [], [], 0
			for word_tag_len in batch:
				no_lower_sentences, sentences, tags, lens = word_tag_len[0], word_tag_len[1], word_tag_len[2], word_tag_len[3]
				no_lower_sen_batch.append(no_lower_sentences)
				sen_batch.append(sentences)
				tag_batch.append(tags)
				len_batch.append(lens)
			sen_max_lens = 60
			if self.mode == "test":
				true_max_len = max(len_batch)
				if true_max_len > sen_max_lens:
					sen_max_lens = true_max_len

			for i in range(len(sen_batch)):
				sen_len = len_batch[i]
				if sen_len <= sen_max_lens:
					dis_len = sen_max_lens - sen_len
					sen_batch[i] += [PAD] * dis_len
					no_lower_sen_batch[i] += [PAD] * dis_len
					tag_batch[i] += ["O"] * dis_len

				else:
					sen_batch[i] = sen_batch[i][:sen_max_lens]
					no_lower_sen_batch[i] = no_lower_sen_batch[i][:sen_max_lens]
					tag_batch[i] = tag_batch[i][:sen_max_lens]
					len_batch[i] = sen_max_lens
			char_batch = self.gen_char(no_lower_sen_batch, sen_max_lens)
			yield sen_batch, char_batch, tag_batch, len_batch, sen_max_lens



	# 加载全部文本数据
	def load_all_data(self):
		datas = []
		words = []
		tags = []
		no_lower_words = []
		for line in codecs.open(self.file, encoding='utf-8'):
			line = line.rstrip().replace('\t', ' ')
			if (len(line) == 0 or line.startswith("-DOCSTART-")):
				if len(words) > 0:
					datas.append((no_lower_words, words, tags, len(words)))
					words = []
					tags = []
					no_lower_words = []
			else:
				word_tag = zero_digits(line).split()
				no_lower_words.append(word_tag[0])
				words.append(word_tag[0].lower())
				tags.append(word_tag[-1])
		return datas

	def get_all_batch(self):
		need_batch_data = self.sort_datas if self.sort else self.datas
		total_batch_num = len(need_batch_data) // self.batch_size
		all_batch = []
		for i in range(total_batch_num):
			start = i * self.batch_size
			end = i * self.batch_size + self.batch_size
			all_batch.append(deepcopy(need_batch_data[start: end]))
		if len(need_batch_data) > end:
			all_batch.append(deepcopy(need_batch_data[end:]))
		return all_batch


	def gen_char(self, sentences, max_len):

		def get_char(sen):
			chars = []
			for word in sen:
				if word != "<PAD>":
					char = [c for c in word[:self.word_max_length]]
					if len(char) < self.word_max_length:
						char += [PAD] * (self.word_max_length - len(char))
				else:
					char = [PAD] * self.word_max_length
				chars.append(char)
			return np.array(chars)

		char_num = np.array([get_char(sen) for sen in sentences])
		char_num = char_num.reshape(-1, max_len * self.word_max_length)
		return char_num




def generate_word(filenames, vocab_path):
	dico = {}
	for i in range(len(filenames)):
		datasets = Datasets(filenames[i], 64, 'test', 20, False)
		for sen_batch, char_batch, tag_batch, len_batch, sen_max_lens in datasets:
			for sen in sen_batch:
				for w in sen:
					if w not in dico:
						dico[w] = 1
					else:
						dico[w] += 1
	sorted_items = sorted(dico.items(), key=lambda x: -x[1])
	for k, v in sorted_items:
			codecs.open(vocab_path, mode='a', encoding='utf-8').write(k + "\n")

if __name__ == '__main__':
	# train_iter = train_iterator('./data/conll2003/train/train_bioes', 64, "train", 2)
	# train_data = train_iter.get_next()
	# configProto = tf.ConfigProto()
	# configProto.gpu_options.visible_device_list = "0"
	# configProto.gpu_options.allow_growth = True
	# with tf.Session(config=configProto) as sess:
	# 	sess.run(train_iter.initializer)
	# 	index = 0
	# 	while True:
	# 		try:
	# 			data = sess.run(train_data)
	# 			sens = data[0]
	# 			char = data[1]
	# 			tag = data[2]
	# 			seq_len = data[3]
	# 			seq_max_len = data[4]
	# 			print(seq_max_len)
	# 			# print(sens.shape)
	# 			# print(char.shape)
	# 			# print(tag.shape)
	# 			# print(seq_len.shape)
	# 			# print(seq_max_len.shape)
	# 			print("step：" + str(index))
	# 			index += 1
	# 			print("\n")
	# 		except tf.errors.OutOfRangeError:
	# 			print("End of dataset")
	# 			break
	# word_vocab = get_vocab('./data/ontonote-5.0/word.vocab')
	# tag_vocab = get_vocab('./data/ontonote-5.0/tag_bioes.vocab')
	# char_vocab = get_vocab('./data/ontonote-5.0/char.vocab')
	# datasets = Datasets('./data/conll2003/test/test_bioes', 64, 'train', 20, False)
	# valid_iterator(datasets, word_vocab, tag_vocab, char_vocab, './result/test.txt', 'result/test_char.txt')
	generate_word(['./data/ontonote-5.0/train/train_bioes', './data/ontonote-5.0/test/test_bioes'], './data/ontonote-5.0/word.vocab')
