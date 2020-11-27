# coding=utf-8

import configparser

class Config(object):

	def __init__(self, configPath):
		config = configparser.ConfigParser()
		config.read(configPath)
		self.learning_rate = float(config.get('model', 'learning_rate'))
		self.dropout = float(config.get('model', 'dropout'))
		self.word_dropout = float(config.get("model", 'word_dropout'))
		self.filter_size = int(config.get('model', 'filter_size'))
		self.pre_word_emb_path = config.get('model', 'pre_word_emb_path')
		self.word_emb_dim = int(config.get('model', 'word_emb_dim'))
		self.char_emb_dim = int(config.get('model', 'char_emb_dim'))
		self.word_vocab_size = int(config.get('model', 'word_vocab_size'))
		self.char_vocab_size = int(config.get('model', 'char_vocab_size'))
		self.num_filter = int(config.get('model', 'num_filter'))
		self.bilstm_hidden_size = int(config.get("model", 'bilstm_hidden_size'))
		self.word_max_length = int(config.get("model", "word_max_length"))
		self.sent_max_length = int(config.get("model", "sent_max_length"))
		self.optimizer = config.get('model', 'optimizer')
		self.source_tag_class = int(config.get('model', 'source_tag_class'))
		self.target_tag_class = int(config.get('model', 'target_tag_class'))
		self.lamta = float(config.get('model', 'lamta'))
		self.beta = float(config.get('model', 'beta'))
		self.gamma = float(config.get('model', 'gamma'))
		self.sigema = float(config.get('model', 'sigema'))

		self.word_vocab = config.get("run", 'word_vocab')
		self.char_vocab = config.get("run", 'char_vocab')
		self.source_tag_vocab = config.get("run", 'source_tag_vocab')
		self.target_tag_vocab = config.get("run", 'target_tag_vocab')
		self.title = config.get("run", "title")
		# self.tensorboard_log = config.get('run', 'tensorboard_log')
		self.best_model_path = config.get('run', 'best_model_path')
		self.gpu = config.get('run', 'gpu')
		self.batch_size = int(config.get('run', 'batch_size'))
		self.step = int(config.get("run", 'step'))
		self.source_train_path = config.get("run", 'source_train_path')
		self.source_dev_path = config.get("run", 'source_dev_path')
		self.source_test_path = config.get("run", 'source_test_path')
		self.target_train_path = config.get("run", 'target_train_path')
		self.target_dev_path = config.get("run", 'target_dev_path')
		self.target_test_path = config.get("run", 'target_test_path')
		self.result_file = config.get('run', 'result_file')
		self.flag = config.get('run', 'flag')
		self.log_file = config.get('run', 'log_file')

if __name__ == '__main__':
	config = Config('./config/attention_grl_transfer_on_r1_100d.0.5.cfg')
	print(config.learning_rate)