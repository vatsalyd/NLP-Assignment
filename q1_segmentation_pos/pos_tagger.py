from collections import defaultdict
import math

def _default_int_dict():
    return defaultdict(int)

def _default_int():
    return 0

class POSTagger:
    def __init__(self, tags):
        self.tags = tags
        self.tag_to_idx = {tag: i for i, tag in enumerate(tags)}
        self.idx_to_tag = {i: tag for i, tag in enumerate(tags)}
        self.num_tags = len(tags)
        
        self.emission_counts = defaultdict(_default_int_dict)
        self.transition_counts = defaultdict(_default_int_dict)
        self.start_counts = defaultdict(_default_int)
        self.second_start_counts = defaultdict(_default_int)
        self.total_emissions = defaultdict(_default_int)
        self.total_transitions = defaultdict(_default_int)
        self._emission_vocab_size = 0

    def train(self, tagged_sentences):
        for sent in tagged_sentences:
            if not sent:
                continue
            prev2_tag = '<START>'
            prev_tag = '<START>'
            for word, tag in sent:
                word = word.lower()
                self.emission_counts[tag][word] += 1
                self.total_emissions[tag] += 1
                
                self.transition_counts[(prev2_tag, prev_tag)][tag] += 1
                self.total_transitions[(prev2_tag, prev_tag)] += 1
                
                if prev2_tag == '<START>':
                    self.start_counts[tag] += 1
                elif prev_tag == '<START>':
                    self.second_start_counts[(prev2_tag, tag)] += 1
                
                prev2_tag, prev_tag = prev_tag, tag

        self._emission_vocab_size = sum(len(v) for v in self.emission_counts.values())

    def emission_prob(self, tag, word, k=1.0):
        return (self.emission_counts[tag].get(word, 0) + k) / (self.total_emissions[tag] + k * self._emission_vocab_size)
    
    def transition_prob(self, prev2_tag, prev_tag, tag, k=1.0):
        if prev2_tag == '<START>' and prev_tag == '<START>':
            return (self.start_counts[tag] + k) / (sum(self.start_counts.values()) + k * self.num_tags)
        elif prev2_tag == '<START>':
            return (self.second_start_counts.get((prev_tag, tag), 0) + k) / (sum(self.second_start_counts.values()) + k * self.num_tags)
        else:
            return (self.transition_counts[(prev2_tag, prev_tag)].get(tag, 0) + k) / (self.total_transitions[(prev2_tag, prev_tag)] + k * self.num_tags)
    
    def viterbi_decode(self, words, k=1.0):
        n = len(words)
        if n == 0:
            return []
        
        dp = [[float('-inf')] * self.num_tags for _ in range(n)]
        backtrack = [[-1] * self.num_tags for _ in range(n)]
        
        for t_idx, tag in enumerate(self.tags):
            trans_prob = self.transition_prob('<START>', '<START>', tag, k)
            emit_prob = self.emission_prob(tag, words[0], k)
            if trans_prob > 0 and emit_prob > 0:
                dp[0][t_idx] = math.log(trans_prob) + math.log(emit_prob)
        
        for i in range(1, n):
            for t_idx, tag in enumerate(self.tags):
                emit_prob = self.emission_prob(tag, words[i], k)
                if emit_prob == 0:
                    continue
                best_prev = -1
                best_score = float('-inf')
                for pt_idx, prev_tag in enumerate(self.tags):
                    if i == 1:
                        trans_prob = self.transition_prob('<START>', prev_tag, tag, k)
                    else:
                        prev2_tag = self.tags[backtrack[i-1][pt_idx]] if backtrack[i-1][pt_idx] != -1 else '<START>'
                        trans_prob = self.transition_prob(prev2_tag, prev_tag, tag, k)
                    if trans_prob > 0:
                        score = dp[i-1][pt_idx] + math.log(trans_prob)
                        if score > best_score:
                            best_score = score
                            best_prev = pt_idx
                if best_prev != -1:
                    dp[i][t_idx] = best_score + math.log(emit_prob)
                    backtrack[i][t_idx] = best_prev
        
        best_last = max(range(self.num_tags), key=lambda t: dp[n-1][t])
        tags_seq = [best_last]
        for i in range(n-1, 0, -1):
            tags_seq.append(backtrack[i][tags_seq[-1]])
        tags_seq.reverse()
        
        return [self.idx_to_tag[t] for t in tags_seq]

def load_ud_spanish_morph(train_path, dev_path, test_path):
    def parse_conllu(filepath):
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            sent = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if sent:
                        sentences.append(sent)
                        sent = []
                    continue
                parts = line.split('\t')
                if len(parts) >= 6 and parts[0].isdigit():
                    word = parts[1]
                    upos = parts[3]
                    feats = parts[5]
                    if feats != '_':
                        morph_tag = f"{upos}"
                        if 'Gender=Fem' in feats:
                            morph_tag += '-Fem'
                        elif 'Gender=Masc' in feats:
                            morph_tag += '-Masc'
                        if 'Number=Sing' in feats:
                            morph_tag += '-Sing'
                        elif 'Number=Plur' in feats:
                            morph_tag += '-Plur'
                        tag = morph_tag
                    else:
                        tag = upos
                    sent.append((word, tag))
            if sent:
                sentences.append(sent)
        return sentences
    
    train_sents = parse_conllu(train_path)
    dev_sents = parse_conllu(dev_path)
    test_sents = parse_conllu(test_path)
    return train_sents, dev_sents, test_sents