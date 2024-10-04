import torch
import torch.nn as nn
import torch.nn.functional as F

#
# DepProbe
#


class DepProbe(nn.Module):
	def __init__(self, emb_model, dep_dim, dep_rels):
		super(DepProbe, self).__init__()
		# internal variables
		self._root_label = dep_rels.index('root')
		# internal models
		self._emb = emb_model
		self._arc = RootedGraphPredictor(self._emb.emb_dim, dep_dim)
		self._lbl = LabelClassifier(self._emb.emb_dim, len(dep_rels), self._root_label)

	def __repr__(self):
		return \
			f'{self.__class__.__name__}:\n' \
			f'  {self._emb}\n' \
			f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>\n' \
			f'  <{self._lbl.__class__.__name__}: {self._lbl._in_dim} -> {self._lbl._num_labels}>'

	def get_trainable_parameters(self):
		return list(self._arc.parameters()) + list(self._lbl.parameters())

	def train(self, mode=True):
		super(DepProbe, self).train(mode)
		self._emb.eval()
		return self

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length)
		# -> ([(batch_size, max_length, emb_dim) * 2], (batch_size, max_length))
		# -> ([emb_sentences_lay0, emb_sentences_lay1], att_sentences)
		with torch.no_grad():
			emb_layers, att_sentences = self._emb(sentences)

		# calculate distances in dependency space
		# dep_embeddings: (batch_size, dep_dim)
		# distances: (batch_size, max_len, max_len)
		dep_embeddings, distances = self._arc(emb_layers[0].detach())

		# classify dependency relations
		lbl_logits = self._lbl(emb_layers[1].detach(), att_sentences.detach())

		# construct minimal return set
		results = {
			'dependency_embeddings': dep_embeddings,
			'distances': distances,
			'label_logits': lbl_logits
		}

		# decode labelled dependency graph
		if decode:
			# get roots and labels from logits
			roots, labels = self._lbl.get_labels(lbl_logits.detach())
			# construct MST starting at root
			graphs = self._arc.to_graph(roots.detach(), distances.detach(), att_sentences.detach())

			# add labels and graphs to results
			results['graphs'] = graphs
			results['labels'] = labels

		return results


class DepProbeMix(DepProbe):
	def __init__(self, emb_model, dep_dim, dep_rels):
		super().__init__(emb_model=emb_model, dep_dim=dep_dim, dep_rels=dep_rels)
		# initialize mixture weights as standard average over layers
		self._mix_arc = nn.Parameter(torch.ones(len(self._emb._lm_layers)))
		self._mix_lbl = nn.Parameter(torch.ones(len(self._emb._lm_layers)))
		self._softmax = nn.Softmax(dim=0)

	def get_trainable_parameters(self):
		return list(self._arc.parameters()) + list(self._lbl.parameters()) + [self._mix_arc, self._mix_lbl]

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length)
		# -> ([(batch_size, max_length, emb_dim) * 2], (batch_size, max_length))
		# -> ([emb_sentences_lay0, emb_sentences_lay1], att_sentences)
		with torch.no_grad():
			emb_layers, att_sentences = self._emb(sentences)

		# compute weighted sum over embedding layers
		emb_sentences_arc = torch.zeros_like(emb_layers[0])
		emb_sentences_lbl = torch.zeros_like(emb_layers[0])
		for layer_idx in range(len(emb_layers)):
			emb_sentences_arc += self._softmax(self._mix_arc)[layer_idx] * emb_layers[layer_idx].detach()
			emb_sentences_lbl += self._softmax(self._mix_lbl)[layer_idx] * emb_layers[layer_idx].detach()

		# calculate distances in dependency space
		# dep_embeddings: (batch_size, dep_dim)
		# distances: (batch_size, max_len, max_len)
		dep_embeddings, distances = self._arc(emb_sentences_arc)

		# classify dependency relations
		lbl_logits = self._lbl(emb_sentences_lbl, att_sentences.detach())

		# construct minimal return set
		results = {
			'dependency_embeddings': dep_embeddings,
			'distances': distances,
			'label_logits': lbl_logits
		}

		# decode labelled dependency graph
		if decode:
			# get roots and labels from logits
			roots, labels = self._lbl.get_labels(lbl_logits.detach())
			# construct MST starting at root
			graphs = self._arc.to_graph(roots.detach(), distances.detach(), att_sentences.detach())

			# add labels and graphs to results
			results['graphs'] = graphs
			results['labels'] = labels

		return results
	

class RetroProbe(nn.Module):
	def __init__(self, emb_model, dep_dim, dep_rels):
		super(RetroProbe, self).__init__()
		# internal variables
		self._root_label = dep_rels.index('root')
		# internal models
		self._emb = emb_model
		self._arc = RootedGraphPredictor(self._emb.emb_dim, dep_dim)
		self._lbl = DependencyLabelClassifier(self._emb.emb_dim*2, len(dep_rels), self._root_label)

	def __repr__(self):
		return \
			f'{self.__class__.__name__}:\n' \
			f'  {self._emb}\n' \
			f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>\n' \
			f'  <{self._lbl.__class__.__name__}: {self._lbl._in_dim} -> {self._lbl._num_labels}>'

	def get_trainable_parameters(self):
		return list(self._arc.parameters()) + list(self._lbl.parameters())

	def train(self, mode=True):
		super(RetroProbe, self).train(mode)
		self._emb.eval()
		return self
	
	def logsumexp(self, x):
		c = torch.max(x, dim=-1).values
		d = torch.sub(x, torch.unsqueeze(c, x.dim()-1))
		return c + torch.logsumexp(d, dim=-1)

	def forward(self, sentences, decode=True):
		# embed sentences (batch_size, seq_length)
		# -> ([(batch_size, max_length, emb_dim) * 2], (batch_size, max_length))
		# -> ([emb_sentences_lay0, emb_sentences_lay1], att_sentences)
		with torch.no_grad():
			emb_layers, att_sentences = self._emb(sentences)

		# calculate distances in dependency space
		# dep_embeddings: (batch_size, dep_dim)
		# distances: (batch_size, max_len, max_len)
		dep_embeddings, distances = self._arc(emb_layers[0].detach())
# 		print(distances)
		parent_probs = F.softmin(distances, dim=-1)
		parent_probs = parent_probs.type(torch.double)
# 		print(parent_probs.size())
# 		parent_probs_norm = self.logsumexp(parent_probs)
# 		parent_probs_norm = torch.unsqueeze(parent_probs_norm,2)
# 		parent_probs = torch.exp(parent_probs-parent_probs_norm)
# 		parent_probs = torch.logsumexp(parent_probs, dim=-1)
# 		print("SOFTMIN")
# 		print(parent_probs)

		# classify dependency relations
		lbl_logits = self._lbl(emb_layers[1].detach(), att_sentences.detach())
# 		print(lbl_logits.size())
		max_size = att_sentences.size(dim=1)
		lbl_logits = torch.reshape(lbl_logits, (lbl_logits.size(dim=0), max_size, max_size, lbl_logits.size(dim=-1)))
		lbl_logits = lbl_logits.type(torch.double)
# 		print(lbl_logits.size())
# 		lbl_parent_logits = torch.zeros(lbl_logits.size(dim=0), max_size, max_size, lbl_logits.size(dim=-1),dtype=torch.double, device=lbl_logits.device)
# 		print(lbl_logits.size())
# 		print(parent_probs.size())
# 		lbl_logits = torch.logsumexp(lbl_logits, dim=-1)
# 		lbl_logits_norm = self.logsumexp(lbl_logits)
# 		lbl_logits_norm = torch.unsqueeze(lbl_logits_norm, 3)
# 		lbl_logits = torch.exp(lbl_logits - lbl_logits_norm)
		
	
		# lbllogits      B x child x Parent x label_probs
		# parent_probs   B x child x Parent
		
		B, child, parent, label_probs = lbl_logits.shape
# 		print("lbl logits shape")
# 		print(lbl_logits.shape)
# 		print(lbl_logits)
		lbl_logits_f = torch.reshape(lbl_logits, (-1, parent, label_probs))
		parent_probs_f = torch.reshape(parent_probs, (-1, parent))
		
		# lbl_logits_f      [B x child] x Parent x label_probs
		# parent_probs_f    [B x child] x Parent
		
		parent_probs_f_u = torch.unsqueeze(parent_probs_f, -1)
		
		# parent_probs_f_u  [B x child] x Parent x 1
		
		
# 		result = self.logsumexp(lbl_logits_f * parent_probs_f_u)
# 		print(result.shape)
# 		print("ligits")
# 		print(lbl_logits_f * parent_probs_f_u)
# 		result = torch.sum(lbl_logits_f * parent_probs_f_u, dim=1) #use log sum trick when multiplying
# 		result = torch.logsumexp(lbl_logits_f + parent_probs_f_u, dim=1)
		result = torch.sum(torch.logaddexp(lbl_logits_f, parent_probs_f_u), dim=1)
		
# 		result = [B x Child] x [label_probs]
		
		result_u = result.reshape((B, child, -1))
# 		print("result shape")
# 		print(result_u.shape)
# 		print(result_u)
		

		#parent_probs = torch.unsqueeze(parent_probs, 3)
		#lbl_parent_logits = torch.logaddexp(lbl_logits, parent_probs)
		#lbl_parent_logits_norm = self.logsumexp(lbl_parent_logits)
		#lbl_parent_logits_norm = torch.unsqueeze(lbl_parent_logits_norm, 3)
		#lbl_parent_logits = torch.exp(lbl_parent_logits - lbl_parent_logits_norm)
 		

		#for i in range(lbl_logits.size(dim=0)):
# 			for j in range(max_size):
# 				for k in range(max_size):
# 					lbl_parent_logits[i,j,k] = torch.add(lbl_logits[i,j,k], parent_probs[i,j,k])
# 		print(lbl_parent_logits)

		# construct minimal return set
		results = {
			'dependency_embeddings': dep_embeddings,
			'distances': distances,
			'label_logits': result_u #lbl_parent_logits
		}

		# decode labelled dependency graph
		if decode:
			# get roots and labels from logits
# 			roots, labels = self._lbl.get_labels(lbl_parent_logits.detach())
			roots, labels = self._lbl.get_labels(result_u.detach())
			# construct MST starting at root
			graphs = self._arc.to_graph(roots.detach(), distances.detach(), att_sentences.detach())

			# add labels and graphs to results
			results['graphs'] = graphs
			results['labels'] = labels

		return results

#
# Graph Predictors
#


class RootedGraphPredictor(nn.Module):
	def __init__(self, embedding_dim, output_dim):
		super(RootedGraphPredictor, self).__init__()
		self._emb_dim = embedding_dim
		self._out_dim = output_dim
		# trainable parameters
		self._transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)

	def forward(self, emb_sentences):
		dep_embeddings = self._transform(emb_sentences)
		batch_size, max_len, out_dim = dep_embeddings.size()

		# calculate differences
		dup_transformed = dep_embeddings.unsqueeze(2)
		dup_transformed = dup_transformed.expand(-1, -1, max_len, -1)
		dup_transposed = dup_transformed.transpose(1, 2)
		differences = dup_transformed - dup_transposed  # (batch_size, max_len, max_len, dep_dim)
		squared_diffs = differences.pow(2)
		distances = torch.sum(squared_diffs, -1)

		return dep_embeddings, distances

	def to_graph(self, roots, distances, mask):
		graphs = torch.ones_like(mask, dtype=torch.long) * -2  # (batch_size, max_len)

		# iterate over sentences
		for sidx in range(graphs.shape[0]):
			# get current sentence length
			sen_len = int(torch.sum(mask[sidx]))

			# set root node's head to -1
			sen_root = int(roots[sidx].detach())
			graphs[sidx, sen_root] = -1

			# gather initial nodes
			tree_nodes = [sen_root]
			free_nodes = [n for n in range(sen_len) if n != sen_root]

			# while there are free nodes, keep adding to graph
			while free_nodes:
				# look for minimum distance between tree and free nodes
				cur_tree_dists = distances[sidx, tree_nodes, :]  # (num_tree_nodes, max_len)
				cur_dists = cur_tree_dists[:, free_nodes]  # (num_tree_nodes, num_free_nodes)
				min_dist_idx = torch.argmin(cur_dists)  # returns argmin of flattened distances # returns tree node, free node
				min_tree = tree_nodes[min_dist_idx // len(free_nodes)]  # tree node of minimum distance pair
				min_free = free_nodes[min_dist_idx % len(free_nodes)]  # free node of minimum distance pair

				# set head node of free node to tree node (point towards root)
				graphs[sidx, min_free] = min_tree

				# housekeeping
				tree_nodes.append(min_free)
				free_nodes.remove(min_free)

		return graphs


#
# Label Predictors
#


class DependencyLabelClassifier(nn.Module):
	def __init__(self, input_dim, num_labels, root_label):
		super(DependencyLabelClassifier, self).__init__()
		self._in_dim = input_dim
		self._num_labels = num_labels  # number of labels (e.g. 37)
		self._root_label = root_label  # index of root label
		# trainable parameters
		self._mlp = nn.Linear(self._in_dim, self._num_labels, bias=False)

	def forward(self, emb_sentences, att_sentences):
		# logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
		# pass through MLP
		att_sentences_expanded = torch.zeros(
			(att_sentences.shape[0], int((att_sentences.shape[1] * (att_sentences.shape[1])))),dtype=torch.bool,
			device=emb_sentences.device
		) 
		for i in range(att_sentences.size()[0]):
			att_size = att_sentences.size()[1]
			for j in range(att_size):
				for k in range(att_size):
					if att_sentences[i][j].item() is True and att_sentences[i][k].item() is True and j!=k:
						att_sentences_expanded[i][(j*att_size)+k] = True
		logits = torch.ones(
			(att_sentences_expanded.shape[0], att_sentences_expanded.shape[1], self._num_labels),
			device=emb_sentences.device
		) * float('-inf')
		
# 		get token embeddings of all sentences (total_tokens, emb_dim)
		new_size = int((emb_sentences.size()[1]*(emb_sentences.size()[1])))
		emb_sentences_expanded = torch.zeros((emb_sentences.size()[0], new_size, emb_sentences.size()[2]*2), device=emb_sentences.device)
# 		k = 0
		start_token = torch.ones_like(emb_sentences[0][0])
		max_len = emb_sentences.size(dim=1)
# 		print(emb_sentences.shape)
# 		print(max_len)
		for i in range(max_len):
			start_idx = i * max_len
			end_idx = start_idx + max_len
			to_cat = torch.clone(emb_sentences[:,i])
			to_cat = to_cat.repeat(1, max_len)
			to_cat = torch.reshape(to_cat, (emb_sentences.size(dim=0), emb_sentences.size(dim=1), emb_sentences.size(dim=2)))
			to_cat[:,i] = torch.clone(start_token)
			cat = torch.cat((emb_sentences, to_cat), 2)
# 			print(cat.shape)
			emb_sentences_expanded[:,start_idx:end_idx] = cat
		emb_words = emb_sentences_expanded[att_sentences_expanded, :]
		# [batch_size] x [combinations (parent, child) ] x [num_labels]
		logits[att_sentences_expanded, :] = self._mlp(emb_words)  # (num_words, num_labels) -> (batch_size, max_len, num_labels)
		# logits are [batch_size] x [combinations (parent, child) ] x [num_labels]
		return logits

	def get_labels(self, lbl_logits):
		# gather word with highest root probability for each sentence
		#logits [batch size] x [child] x [parent] x [num labels]
# 		logits now of batch size x child x numlabels

		lbl_logits = F.softmax(lbl_logits, dim=-1)
		lbl_logits = torch.nan_to_num(lbl_logits, nan=float('-inf'))
		roots = torch.argmax(lbl_logits[:, :, self._root_label], dim=-1)  # (batch_size, 1)
		# set root logits to -inf to prevent multiple roots
		lbl_logits_noroot = lbl_logits.detach().clone()
		lbl_logits_noroot[:, :, self._root_label] = torch.ones(
			(lbl_logits.shape[0], lbl_logits.shape[1]),
			device=lbl_logits.device
		) * float('-inf')
		# get predicted labels with maximum probability (padding should have -inf)
		labels = torch.argmax(lbl_logits_noroot, dim=-1)  # (batch_size, max_len)
		# add true root labels
		labels[torch.arange(lbl_logits.shape[0]), roots] = self._root_label
		# add -1 padding
		labels[(lbl_logits[:, :, 0] == float('-inf'))] = -1

		
# 		lbl_logits = F.softmax(lbl_logits, dim=-1)
# 		num_labels = lbl_logits.size(dim=3)

# 		#This maxes over child
# 		roots = torch.max(lbl_logits[:, :,:, self._root_label], dim=-2)  # (batch_size, parent, 1)
		
# 		roots_max = torch.max(roots.values, dim=-1) # (batch_size, 1)
	
	
# 		roots_final = roots.indices[:,roots_max.indices[:]]
# 		roots_final = torch.diagonal(roots_final)
# 		roots_final = torch.unsqueeze(roots_final,1)
# 		# set root logits to -inf to prevent multiple roots
# 		lbl_logits_noroot = lbl_logits.detach().clone()
# 		lbl_logits_noroot[:, :, :, self._root_label] = torch.ones(
# 			(lbl_logits.shape[0], lbl_logits.shape[1], lbl_logits.shape[2]),
# 			device=lbl_logits.device
# 		) * float('-inf')
		
# 		pred_label_logits = torch.transpose(lbl_logits_noroot, 2, 3)
# 		#                                                                                           child,   parent*num_labels 
# 		pred_label_logits = torch.flatten(lbl_logits_noroot, start_dim=2, end_dim=3) # (batch_size, max_len, max_len*num_labels)
		
# 		# get predicted labels with maximum probability (padding should have -inf)
# 		# add true root labels
# 		labels = torch.argmax(pred_label_logits,dim=-1) % (num_labels)
# 		labels[torch.arange(lbl_logits.shape[0]), roots_final.flatten()] = self._root_label
# 		# add -1 padding
# 		labels[(pred_label_logits[:, :, 0] == float('-inf'))%(num_labels)] = -1

		return roots, labels
	
class LabelClassifier(nn.Module):
	def __init__(self, input_dim, num_labels, root_label):
		super(LabelClassifier, self).__init__()
		self._in_dim = input_dim
		self._num_labels = num_labels  # number of labels (e.g. 37)
		self._root_label = root_label  # index of root label
		# trainable parameters
		self._mlp = nn.Linear(self._in_dim, self._num_labels, bias=False)

	def forward(self, emb_sentences, att_sentences):
		# logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
		logits = torch.ones(
			(att_sentences.shape[0], att_sentences.shape[1], self._num_labels),
			device=emb_sentences.device
		) * float('-inf')
		print("emb sentences size")
		print(emb_sentences.size())
		# get token embeddings of all sentences (total_tokens, emb_dim)
		emb_words = emb_sentences[att_sentences, :]
		print("emb words size")
		print(emb_words.size())
		
		# pass through MLP
		logits[att_sentences, :] = self._mlp(emb_words)  # (num_words, num_labels) -> (batch_size, max_len, num_labels)
		return logits

	def get_labels(self, lbl_logits):
		# gather word with highest root probability for each sentence
# 		print(lbl_logits)
		lbl_logits = F.softmax(lbl_logits, dim=-1)
		lbl_logits = torch.nan_to_num(lbl_logits, nan=float('-inf'))
# 		print(lbl_logits)
		roots = torch.argmax(lbl_logits[:, :, self._root_label], dim=-1)  # (batch_size, 1)
		# set root logits to -inf to prevent multiple roots
		lbl_logits_noroot = lbl_logits.detach().clone()
		lbl_logits_noroot[:, :, self._root_label] = torch.ones(
			(lbl_logits.shape[0], lbl_logits.shape[1]),
			device=lbl_logits.device
		) * float('-inf')
		# get predicted labels with maximum probability (padding should have -inf)
		labels = torch.argmax(lbl_logits_noroot, dim=-1)  # (batch_size, max_len)
		# add true root labels
		labels[torch.arange(lbl_logits.shape[0]), roots] = self._root_label
		# add -1 padding
		labels[(lbl_logits[:, :, 0] == float('-inf'))] = -1

		return roots, labels
