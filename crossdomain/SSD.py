# coding=utf-8
import tensorflow as tf
from crossdomain.DataLoader import  *
from tensorflow.contrib.rnn import LSTMCell
from crossdomain.config import Config
import argparse
from crossdomain.conlleval import evaluate
from tabulate import tabulate
from numpy.random import seed
from tensorflow import set_random_seed
import random
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import os

class Network(object):

	def __init__(self, config):
		self.config = config

	def inference(self, word_input, char_input, tag_input, sen_len, train_mode, tag_class, domain_input, is_reuse,
	              crf_scope, fea_reuse, word_embedding, char_embedding, batch_size, sen_max_len):
		word_embedding_label = tf.nn.embedding_lookup(word_embedding, word_input)
		word_emb = tf.layers.dropout(word_embedding_label, self.config.word_dropout, training=train_mode)
		char_emb = tf.nn.embedding_lookup(char_embedding, char_input)
		char_embed_dropout = tf.layers.dropout(char_emb, self.config.dropout, training=train_mode, name="char_embed_dropout")
		str_feature = self.feature_extractor(train_mode, sen_len, fea_reuse, word_emb, char_embed_dropout, scope_name='str_feature_extractor', batch_size=batch_size, sen_max_len=sen_max_len)
		sem_feature = self.feature_extractor(train_mode, sen_len, fea_reuse, word_emb, char_embed_dropout, scope_name='sem_feature_extractor', batch_size=batch_size, sen_max_len=sen_max_len)
		str_mi_feature = tf.reshape(str_feature, [batch_size * sen_max_len, 2*self.config.bilstm_hidden_size])
		sem_mi_feature = tf.reshape(sem_feature, [batch_size * sen_max_len, 2*self.config.bilstm_hidden_size])
		stru_sem_mi = self.stru_sem_mine(str_mi_feature, sem_mi_feature, sen_max_len, batch_size)
		stru_sem_mi = tf.clip_by_value(stru_sem_mi, -3, 3)
		feature = tf.concat([str_feature, sem_feature], axis=-1)
		with tf.variable_scope(crf_scope, initializer=tf.contrib.layers.xavier_initializer(), reuse=is_reuse):
			logits = tf.layers.dense(feature, tag_class, name="logits")
			log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, tag_input, sen_len)
			pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_params, sen_len)
			loss = tf.reduce_mean(-log_likelihood)

		with tf.variable_scope("decoder", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
			decoder_logits = tf.layers.dense(feature, self.config.word_emb_dim, name="decoder_logits")
			decoder_mi_logits = tf.reshape(decoder_logits, [batch_size * sen_max_len, self.config.word_emb_dim])
			masked = tf.sequence_mask(sen_len, sen_max_len, dtype=tf.float32)
			decoder_loss = tf.losses.mean_squared_error(labels=word_embedding_label, predictions=decoder_logits, reduction=tf.losses.Reduction.NONE)
			decoder_loss = tf.sqrt(tf.reduce_sum(decoder_loss, axis=-1))
			decoder_loss = tf.reduce_sum(decoder_loss * masked, axis=-1)
			decoder_loss = tf.reduce_mean(decoder_loss)

		with tf.variable_scope("domain_predictor", initializer=tf.contrib.layers.xavier_initializer(),
		                       reuse=tf.AUTO_REUSE):
			domain_feature = tf.nn.max_pool(tf.expand_dims(sem_feature, -1)
			                                , [1, self.config.sent_max_length, 1, 1],
			                                [1, self.config.sent_max_length, 1, 1], padding='SAME')
			domain_feature = tf.squeeze(domain_feature, axis=[1, 3])
			domain_logits = tf.layers.dense(domain_feature, 2, name="domain_logits")
			domain_pre_id = tf.argmax(tf.nn.softmax(domain_logits, axis=-1), axis=-1, output_type=tf.int32)
			domain_acc = tf.reduce_mean(tf.cast(tf.equal(domain_pre_id, domain_input), tf.float32))
			domain_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=domain_input, logits=domain_logits)
			domain_loss = tf.reduce_mean(domain_loss)
		str_decoder_mi = self.str_decoder_mine(str_mi_feature, decoder_mi_logits, sen_max_len, batch_size)
		sem_decoder_mi = self.sem_decoder_mine(sem_mi_feature, decoder_mi_logits, sen_max_len, batch_size)
		return pred_ids, loss, decoder_loss, stru_sem_mi, sem_decoder_mi, domain_loss, domain_acc, str_decoder_mi

	def feature_extractor(self, train_mode, sen_len, fea_reuse, word_emb, char_emb, scope_name, batch_size, sen_max_len):
		with tf.variable_scope(scope_name, initializer=tf.contrib.layers.xavier_initializer(),
		                       reuse=fea_reuse):
			filter_shape = [self.config.filter_size, self.config.char_emb_dim, self.config.num_filter]
			# input [batch, length, emb_dim]
			kernel = tf.get_variable(shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer(),
			                         name="kernel")
			conv = tf.nn.conv1d(char_emb, kernel, 1, padding='SAME', name="conv")
			conv_expand = tf.expand_dims(conv, -1)
			# 缺一部分最大池化然后变成和word_input相同输入
			char_pool = tf.nn.max_pool(conv_expand, [1, self.config.word_max_length, 1, 1],
			                           [1, self.config.word_max_length, 1, 1], padding='SAME')
			char_pool_flatten = tf.reshape(char_pool, [batch_size, sen_max_len, self.config.num_filter])
			word_char_feature = tf.concat([word_emb, char_pool_flatten], axis=2)
			word_char_feature_dropout = tf.layers.dropout(word_char_feature, self.config.dropout,
			                                              training=train_mode,
			                                              name="word_char_feature_dropout")

			"""encoder"""
			self.lstm_fw_cell = LSTMCell(self.config.bilstm_hidden_size, use_peepholes=True)
			self.lstm_bw_cell = LSTMCell(self.config.bilstm_hidden_size, use_peepholes=True)
			(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
				self.lstm_fw_cell, self.lstm_bw_cell, word_char_feature_dropout,
				sequence_length=sen_len, dtype=tf.float32)
			enc_outputs = tf.concat([output_fw, output_bw], axis=-1, name="biLstm")
			outputs = self.self_attention(enc_outputs, train_mode)
			outputs = tf.layers.dropout(outputs, self.config.dropout, training=train_mode)

			return outputs

	def normalize(self, inputs, epsilon=1e-8, scope="normalization"):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
			inputs_shape = inputs.get_shape()
			params_shape = inputs_shape[-1:]
			mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
			beta = tf.get_variable(name="beta", shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
			gamma = tf.get_variable(name='gamma', shape=params_shape, dtype=tf.float32, initializer=tf.ones_initializer)
			normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
			outputs = gamma * normalized + beta
			return outputs


	def self_attention(self, keys, train_mode, scope='multihead_attention'):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			Q = tf.nn.relu(
				tf.layers.dense(keys, 2 * self.config.bilstm_hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer()))
			K = tf.nn.relu(
				tf.layers.dense(keys, 2 * self.config.bilstm_hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer()))
			V = tf.nn.relu(
				tf.layers.dense(keys, 2 * self.config.bilstm_hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer()))
			#[batch*num_head, sequnece_length, hidden_size/num_head]
			Q_ = tf.concat(tf.split(Q, self.config.num_heads, axis=2), axis=0)
			K_ = tf.concat(tf.split(K, self.config.num_heads, axis=2), axis=0)
			V_ = tf.concat(tf.split(V, self.config.num_heads, axis=2), axis=0)
			outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
			outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
			key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) #[batch_size, sequence_length]
			key_masks = tf.tile(key_masks, [self.config.num_heads, 1]) #[batch_size*num_head, sequence_length]
			key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
			paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
			outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
			outputs = tf.nn.softmax(outputs)
			query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
			query_masks = tf.tile(query_masks, [self.config.num_heads, 1])
			query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
			outputs *= query_masks
			outputs = tf.layers.dropout(outputs, self.config.dropout, training=train_mode)
			outputs = tf.matmul(outputs, V_)
			outputs = tf.concat(tf.split(outputs, self.config.num_heads, axis=0), axis=2)
			outputs += keys
			outputs = self.normalize(outputs)
		return outputs

	def sem_decoder_mine(self, x, y, sen_max_len, batch_size):
		with tf.variable_scope("sem_docoder_mine", reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
			shuffle = tf.expand_dims(tf.random_shuffle(tf.eye(batch_size * sen_max_len)), -1)
			shuffle = tf.stop_gradient(shuffle)
			y_shuffle = tf.map_fn(lambda s: tf.reduce_sum(s * y, axis=0), elems=shuffle, dtype=tf.float32, back_prop=False)
			t_ab = self.sem_decoder_T(x, y)
			t_a_b = self.sem_decoder_T(x, y_shuffle)
			mi_loss = tf.reduce_mean(t_ab) - tf.log(tf.reduce_mean(tf.exp(t_a_b)))
		return mi_loss

	def sem_decoder_T(self, x, y):
		with tf.variable_scope("sem_decoder_T", reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
			plus_bias = tf.get_variable(name="plus_bias", shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer)
			layer_x = tf.layers.dense(x, 256, use_bias=False)
			layer_y = tf.layers.dense(y, 256, use_bias=False)
			output = tf.nn.leaky_relu(layer_x + layer_y + plus_bias)
			output = tf.tanh(tf.layers.dense(output, 64))
			output = tf.nn.leaky_relu(tf.layers.dense(output, 1))

			return output

	def str_decoder_mine(self, x, y, sen_max_len, batch_size):
		with tf.variable_scope("str_docoder_mine", reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
			shuffle = tf.expand_dims(tf.random_shuffle(tf.eye(batch_size * sen_max_len)), -1)
			shuffle = tf.stop_gradient(shuffle)
			y_shuffle = tf.map_fn(lambda s: tf.reduce_sum(s * y, axis=0), elems=shuffle, dtype=tf.float32, back_prop=False)
			t_ab = self.str_decoder_T(x, y)
			t_a_b = self.str_decoder_T(x, y_shuffle)
			mi_loss = tf.reduce_mean(t_ab) - tf.log(tf.reduce_mean(tf.exp(t_a_b)))
		return mi_loss

	def str_decoder_T(self, x, y):
		with tf.variable_scope("str_decoder_T", reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
			plus_bias = tf.get_variable(name="plus_bias", shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer)
			layer_x = tf.layers.dense(x, 256, use_bias=False)
			layer_y = tf.layers.dense(y, 256, use_bias=False)
			output = tf.nn.leaky_relu(layer_x + layer_y + plus_bias)
			output = tf.tanh(tf.layers.dense(output, 64))
			output = tf.nn.leaky_relu(tf.layers.dense(output, 1))

			return output

	def stru_sem_mine(self, x, y, sen_max_len, batch_size):
		with tf.variable_scope("str_sem_mine", reuse=tf.AUTO_REUSE,initializer=tf.contrib.layers.xavier_initializer()):
			x_ab = tf.concat([x, y], -1)
			shuffle = tf.expand_dims(tf.random_shuffle(tf.eye(batch_size * sen_max_len)), -1)
			shuffle = tf.stop_gradient(shuffle)
			y_shuffle = tf.map_fn(lambda s: tf.reduce_sum(s * y, axis=0), elems=shuffle, dtype=tf.float32, back_prop=False)
			x_a_b = tf.concat([x, y_shuffle], -1)
			t_ab = self.stru_sem_T(x_ab)
			t_a_b = self.stru_sem_T(x_a_b)
			mi_loss = tf.reduce_mean(t_ab) - tf.log(tf.reduce_mean(tf.exp(t_a_b)))
		return mi_loss

	def stru_sem_T(self, input):
		with tf.variable_scope("stru_sem_T", reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
			layer = tf.layers.dense(input, 256, use_bias=True)
			layer3 = tf.tanh(layers.linear(layer, 64))
			output = tf.nn.leaky_relu(layers.linear(layer3, 1))
		return output



def log(message, config):
	if config.flag == "file":
		result_file_name = config.log_file
		log_file = result_file_name
		codecs.open(log_file, mode='a', encoding='utf-8').write(message + "\n")
	else:
		print(message)


def draw(x, y, path):
	fig, ax = plt.subplots()
	ax.plot(x, y, label='MINE estimate')
	ax.set_xlabel('training steps')
	fig.savefig(path)
	plt.close()


if __name__ == '__main__':
	"""定义数据集以及一些必须文件"""
	parser = argparse.ArgumentParser(description="run ner-model")
	parser.add_argument("--config_path", type=str, required=True)
	args = parser.parse_args()
	config_path = args.config_path
	config = Config(config_path)
	# config.flag = "console"
	config.num_heads = 8
	config.gan_epoch = 3
	word_vocab = get_vocab(config.word_vocab)
	char_vocab = get_vocab(config.char_vocab)
	source_tag_vocab = get_vocab(config.source_tag_vocab)
	target_tag_vocab = get_vocab(config.target_tag_vocab)
	idx_to_tag = {idx: tag for tag, idx in target_tag_vocab.items()}
	idx_to_word = {idx: tag for tag, idx in word_vocab.items()}
	test = Datasets(config.target_test_path, config.batch_size, 'test', config.word_max_length, False)
	source_train = Datasets(config.source_train_path, config.batch_size, 'train', config.word_max_length, False)
	target_train = Datasets(config.target_train_path, config.batch_size, 'train', config.word_max_length, False)
	source_batch_generator = dsr_train_batch(source_train, word_vocab, source_tag_vocab, char_vocab, "source")
	target_batch_generator = dsr_train_batch(target_train, word_vocab, target_tag_vocab, char_vocab, "target")
	configProto = tf.ConfigProto()
	configProto.gpu_options.visible_device_list = config.gpu
	configProto.gpu_options.allow_growth = True
	gan_step = 0
	best_test_f1Score = 0
	best_f1_p = 0
	best_f1_r = 0
	best_gan_epoch = 0
	best_step = 0
	g = tf.Graph()
	model = Network(config)
	random.seed(88)
	seed(88)
	with g.as_default():
		set_random_seed(88)
		source_word_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="source_word_input")
		source_char_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="source_char_input")
		source_tag_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="source_tag_input")
		source_sent_len = tf.placeholder(dtype=tf.int32, shape=[None], name="source_sent_len")
		source_domain_input = tf.placeholder(dtype=tf.int32, shape=[None], name="source_domain_input")
		source_sen_max_len = tf.placeholder(dtype=tf.int32, shape=(), name="source_sen_max_len")

		target_word_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="target_word_input")
		target_char_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="target_char_input")
		target_tag_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="target_tag_input")
		target_sent_len = tf.placeholder(dtype=tf.int32, shape=[None], name="target_sent_len")
		target_domain_input = tf.placeholder(dtype=tf.int32, shape=[None], name="target_domain_input")
		target_sen_max_len = tf.placeholder(dtype=tf.int32, shape=(), name="target_sen_max_len")

		train_mode = tf.placeholder(dtype=tf.bool, shape=(), name="train_mode")
		source_batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="source_batch_size")
		target_batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="target_batch_size")

		# word_embedding = tf.Variable(
		# 	tf.constant(0.0, dtype=tf.float32, shape=[config.word_vocab_size, config.word_emb_dim]),
		# 	name="word_embed")

		word_embedding = tf.Variable(np.load(config.pre_word_emb_path)['embeddings'], name="word_embed", dtype=tf.float32)

		char_embedding = tf.Variable(
			tf.random_uniform([config.char_vocab_size, config.char_emb_dim],
			                  -np.sqrt(3.0 / config.char_emb_dim),
			                  np.sqrt(3.0 / config.char_emb_dim)), name="char_emb", dtype=tf.float32)

		source_pred_ids, source_loss, source_decoder_loss, source_stru_sem_mi, source_sem_decoder_mi, source_domain_loss, source_domain_acc, source_str_decoder_mi = model.inference(
			source_word_input, source_char_input, source_tag_input,
			source_sent_len, train_mode, config.source_tag_class, source_domain_input, False, "source_crf",
			fea_reuse=False, word_embedding=word_embedding, char_embedding=char_embedding, batch_size=source_batch_size, sen_max_len=source_sen_max_len)
		target_pred_ids, target_loss, target_decoder_loss, target_stru_sem_mi, target_sem_decoder_mi, target_domain_loss, target_domain_acc, target_str_decoder_mi = model.inference(
			target_word_input, target_char_input, target_tag_input,
			target_sent_len, train_mode, config.target_tag_class, target_domain_input, False, "target_crf",
			fea_reuse=True, word_embedding=word_embedding, char_embedding=char_embedding, batch_size=target_batch_size, sen_max_len=target_sen_max_len)
		if config.optimizer == "adam":
			optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
		elif config.optimizer == "sgd":
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)

		training_variables = tf.trainable_variables()
		dsr_variable = [v for v in training_variables if "mine" not in v.name]
		mine_variable = [v for v in training_variables if "mine" in v.name]
		ssd_loss = source_loss + target_loss + config.lamta * (source_decoder_loss + target_decoder_loss) + config.beta * (source_domain_loss + target_domain_loss)
		ssd_mi_loss = source_stru_sem_mi + target_stru_sem_mi - source_sem_decoder_mi - target_sem_decoder_mi - source_str_decoder_mi - target_str_decoder_mi
		ssd_total_loss = ssd_loss + config.gamma * ssd_mi_loss

		#更新ssd网络参数
		ssd_op = tf.train.AdamOptimizer(config.learning_rate)
		print("---------------ssd框架需要更新的参数--------------")
		for v in dsr_variable:
			print(v)
		print("-------------------------------------------------")
		ssd_gradients = ssd_op.compute_gradients(ssd_total_loss, var_list=dsr_variable)
		ssd_grads = [grad for grad, var in ssd_gradients]
		ssd_train_vars = [var for grad, var in ssd_gradients]
		ssd_grad_clip_list, _ = tf.clip_by_global_norm(ssd_grads, 0.9)
		ssd_train_op = ssd_op.apply_gradients((zip(ssd_grad_clip_list, ssd_train_vars)))

		# 更新mine网络参数
		print("---------------mi框架需要更新的参数--------------")
		for v in mine_variable:
			print(v)
		print("-------------------------------------------------")
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			syn_sem_mi_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			syn_dec_mi_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			sem_dec_mi_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

			syn_sem_mi_loss = -(source_stru_sem_mi + target_stru_sem_mi)
			syn_dec_mi_loss = -(source_str_decoder_mi + target_str_decoder_mi)
			sem_dec_mi_loss = -(source_sem_decoder_mi + target_sem_decoder_mi)

			# 语法语义mi训练
			syn_sem_gradients = syn_sem_mi_optimizer.compute_gradients(syn_sem_mi_loss, var_list=mine_variable)
			syn_sem_grads = [grad for grad, var in syn_sem_gradients]
			syn_sem_train_vars = [var for grad, var in syn_sem_gradients]
			syn_sem_gradient_list, syn_sem_ = tf.clip_by_global_norm(syn_sem_grads, 0.9)
			syn_sem_train_op = syn_sem_mi_optimizer.apply_gradients(zip(syn_sem_gradient_list, syn_sem_train_vars))

			# 语法和重构器mi训练
			syn_dec_gradients = syn_dec_mi_optimizer.compute_gradients(syn_dec_mi_loss, var_list=mine_variable)
			syn_dec_grads = [grad for grad, var in syn_dec_gradients]
			syn_dec_train_vars = [var for grad, var in syn_dec_gradients]
			syn_dec_gradient_list, syn_dec_ = tf.clip_by_global_norm(syn_dec_grads, 0.9)
			syn_dec_train_op = syn_dec_mi_optimizer.apply_gradients(zip(syn_dec_gradient_list, syn_dec_train_vars))

			# 语义和重构器mi训练
			sem_dec_gradients = sem_dec_mi_optimizer.compute_gradients(sem_dec_mi_loss, var_list=mine_variable)
			sem_dec_grads = [grad for grad, var in sem_dec_gradients]
			sem_dec_train_vars = [var for grad, var in sem_dec_gradients]
			sem_dec_gradient_list, sem_dec_ = tf.clip_by_global_norm(sem_dec_grads, 0.9)
			sem_dec_train_op = sem_dec_mi_optimizer.apply_gradients(zip(sem_dec_gradient_list, sem_dec_train_vars))

			mine_train_op = tf.group([syn_sem_train_op, syn_dec_train_op, sem_dec_train_op])

		saver = tf.train.Saver()
		# tbWriter = SummaryWriter(config.tensorboard_log)
		def get_feed_dict(source_word, source_src_len, source_char, source_tag, source_domain_label, target_word,
		                  target_src_len, target_char, target_tag, target_domain_label, source_batch, target_batch, source_max_len, target_max_len):
			feed = {
				source_word_input: source_word,
				source_char_input: source_char,
				source_sent_len: source_src_len,
				source_tag_input: source_tag,
				source_domain_input: source_domain_label,

				target_word_input: target_word,
				target_char_input: target_char,
				target_sent_len: target_src_len,
				target_tag_input: target_tag,
				target_domain_input: target_domain_label,

				train_mode: True,
				source_batch_size: source_batch,
				target_batch_size: target_batch,
				source_sen_max_len: source_max_len,
				target_sen_max_len: target_max_len
			}
			return feed
		with tf.Session(config=configProto) as sess:
			tf.global_variables_initializer().run()
			# 对抗更新
			while gan_step <= config.gan_epoch:
				step = 0
				while step < 2000:
					source_word, source_char, source_tag, source_src_len, source_max_len, source_domain_label = next(source_batch_generator)
					target_word, target_char, target_tag, target_src_len, target_max_len, target_domain_label = next(target_batch_generator)
					feed = get_feed_dict(source_word, source_src_len, source_char, source_tag, source_domain_label,
					                     target_word, target_src_len, target_char, target_tag, target_domain_label,
					                     len(source_word), len(target_word), source_max_len, target_max_len)
					_, a, b, c, d, e, f, g, h, i, j, k, L = sess.run([ssd_train_op, source_stru_sem_mi, target_stru_sem_mi, source_sem_decoder_mi, target_sem_decoder_mi, source_loss, target_loss, source_decoder_loss, target_decoder_loss, source_domain_acc, target_domain_acc, source_str_decoder_mi, target_str_decoder_mi], feed_dict=feed)
					header = ["gan_step", "title", "source_crf_loss", "target_crf_loss", "source_decoder_loss", "target_decoder_loss",
					"source_domain_acc", "target_domain_acc", "source_stru_sem_mine", "target_stru_sem_mine", "source_sem_decoder_mine", "target_sem_decoder_mine", "source_str_decoder_mine", "target_str_decoder_mine"]
					table = [[gan_step, config.title, e, f, g, h, i, j, a, b, c, d, k, L]]
					log("step is : {}".format(step), config)
					log(tabulate(table, headers=header, tablefmt='grid'), config)
					log_step = gan_step * 2000 + step

					if step % 10 == 0:
						gold_label_data = []
						pred_label_data = []
						golds = []
						preds = []
						words = []
						gold_label_num = []
						pred_label_num = []
						all_lens = []
						for test_src_input, test_char_input, test_tag_label, test_src_len, test_max_len in dsr_test_batch(test, word_vocab, target_tag_vocab, char_vocab):
							test_feed = {
								target_word_input: test_src_input,
								target_char_input: test_char_input,
								target_sent_len: test_src_len,
								target_tag_input: test_tag_label,
								train_mode: False,
								target_sen_max_len: test_max_len,
								target_batch_size: len(test_src_input)
							}
							y_preds = sess.run(target_pred_ids, feed_dict=test_feed)
							for word_arr, gold_label, pred_label, length in zip(test_src_input, test_tag_label, y_preds,
							                                                    test_src_len):
								word = word_arr[:length]
								word_str = [idx_to_word[l] for l in word]
								words.append(word_str)
								lab = gold_label[:length]
								lab_pred = pred_label[:length]
								gold_label_num.append(lab)
								pred_label_num.append(lab_pred)
								all_lens.append(length)
								lab_str = [idx_to_tag[l] for l in lab]
								lab_pre_str = [idx_to_tag[l] for l in lab_pred]
								golds.append(lab_str)
								preds.append(lab_pre_str)
								gold_label_data += lab_str
								pred_label_data += lab_pre_str
						test_p, test_r, test_f1 = evaluate(gold_label_data, pred_label_data, False)
						if test_f1 > best_test_f1Score:
							best_gan_epoch = gan_step
							best_step = step
							best_test_f1Score = test_f1
							best_f1_p = test_p
							best_f1_r = test_r
							if os.path.exists(config.result_file):
								# 删除文件，可使用以下两种方法。
								os.remove(config.result_file)
							for word, true, pred in zip(words, golds, preds):
								for w, t, p in zip(word, true, pred):
									codecs.open(config.result_file, 'a', encoding='utf-8').write(w + " " + t + " " + p + '\n')
								codecs.open(config.result_file, 'a', encoding='utf-8').write('\n')
							saver.save(sess, config.best_model_path)
						log("step is : {}".format(step), config)
						table = [['test', gan_step, config.title, test_p, test_r, test_f1, best_gan_epoch, best_step, best_f1_p, best_f1_r, best_test_f1Score]]
						header = ["data", "gan_step", 'title', "test_p", "test_r", 'test_f1', 'best_gan_epoch', 'best_step', 'best_f1_p', 'best_f1_r', 'best_f1']
						log(tabulate(table, headers=header, tablefmt='grid'), config)
					step += 1

				if gan_step == config.gan_epoch:
					step = 150000
				else:
					step = 0

				while step <= 1000:
					source_word, source_char, source_tag, source_src_len, source_max_len, source_domain_label = next(
						source_batch_generator)
					target_word, target_char, target_tag, target_src_len, target_max_len, target_domain_label = next(
						target_batch_generator)
					feed = get_feed_dict(source_word, source_src_len, source_char, source_tag, source_domain_label,
					                     target_word, target_src_len, target_char, target_tag, target_domain_label,
					                     source_batch=len(source_word), target_batch=len(target_word), source_max_len=source_max_len, target_max_len=target_max_len)
					_, a, b, c, d, e, f = sess.run(
						[mine_train_op, source_stru_sem_mi, target_stru_sem_mi, source_sem_decoder_mi,
						 target_sem_decoder_mi, source_str_decoder_mi, target_str_decoder_mi], feed_dict=feed)

					header = ["title", "gan_step", "best_gan_epoch", "best_gan_step", "best_f1_score", "source_stru_sem_mine", "target_stru_sem_mine", "source_sem_decoder_mine", "target_sem_decoder_mine", "source_str_decoder_mine", "target_str_decoder_mine"]
					table = [[config.title+"_mine_train", gan_step, best_gan_epoch, best_step, best_test_f1Score, a, b, c, d, e, f]]
					log("step is : {}".format(step), config)
					log(tabulate(table, headers=header, tablefmt='grid'), config)
					step += 1

				gan_step += 1
