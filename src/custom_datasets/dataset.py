from torch.utils.data import Dataset
from datasets import load_dataset
import os
import json
import torch


class ACEDataset(Dataset):
    def __init__(self, data_dir, split, max_length=256):
        data_path = os.path.join(data_dir, f'{split}.json')
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.data.append(example)

        with open(os.path.join(data_dir, 'label2id.json'), 'r') as f:
            self.label2id = json.load(f)

        self.id2label = {v: k for k, v in self.label2id.items()}

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_ids = torch.tensor(example['input_ids'][:self.max_length])
        attention_mask = torch.tensor(example['attention_mask'][:self.max_length])
        labels = torch.tensor(example['iob_labels'][:self.max_length])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ConllDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length=256):
        dataset = load_dataset(data_dir)[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        self.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        self.tokenized = dataset.map(tokenize_and_align_labels, batched=True)

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        example = self.tokenized[idx]
        if self.max_length > len(example['input_ids']):
            input_ids = torch.tensor(example['input_ids'] + [self.pad_token_id] * (self.max_length - len(example['input_ids'])))
            attention_mask = torch.tensor(example['attention_mask'] + [0] * (self.max_length - len(example['attention_mask'])))
            labels = torch.tensor(example['labels'] + [-100] * (self.max_length - len(example['labels'])))
        else:
            input_ids = torch.tensor(example['input_ids'][:self.max_length - 1] + [example['input_ids'][-1]])
            attention_mask = torch.tensor(example['attention_mask'][:self.max_length])
            labels = torch.tensor(example['labels'][:self.max_length - 1] + [example['labels'][-1]])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    
class JointConllDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length=256):
        dataset = load_dataset(data_dir)[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        self.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        label2label_name = {
            'O': 'other',
            'B-PER': 'person',
            'I-PER': 'person',
            'B-ORG': 'organization',
            'I-ORG': 'organization',
            'B-LOC': 'location',
            'I-LOC': 'location',
            'B-MISC': 'miscellaneous',
            'I-MISC': 'miscellaneous'
        }

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels

            prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode("### Instruction:\nplease extract named entities and their type from the input sentence, all entity types are in options\n### Options:\nperson, location, organization, miscellaneous\n### Sentence:\n")
            prefix_mask = [1] * len(prefix_ids)
            prefix_labels = [-100] * len(prefix_ids)

            middle_ids = tokenizer.encode("\n### Response:\n")
            middle_mask = [1] * len(middle_ids)
            middle_labels = [-100] * len(middle_ids)

            batch_it_labels = []
            debug = []

            for i, label in enumerate(examples[f"ner_tags"]):
                tokens = examples["tokens"][i]
                it_tokens = []
                cur_it_token = {
                    "tokens": [],
                    "label": None
                }
                for j, label_id in enumerate(label):
                    if label_id == 0:
                        if cur_it_token['tokens']:
                            # cur_it_token['tokens'] += [":", cur_it_token['label'], ";"]
                            it_tokens += [f"{' '.join(cur_it_token['tokens'])}:{cur_it_token['label']}"]
                            cur_it_token = {
                                "tokens": [],
                                "label": None
                            }
                    else:
                        label_name = self.id2label[label_id]
                        label_full_name = label2label_name[label_name]

                        if "B-" in label_name:
                            if cur_it_token['tokens']:
                                # cur_it_token['tokens'] += [":", cur_it_token['label'], ";"]
                                it_tokens += [f"{' '.join(cur_it_token['tokens'])}:{cur_it_token['label']}"]
                                cur_it_token = {
                                    "tokens": [],
                                    "label": None
                                }

                        cur_it_token['label'] = label_full_name
                        cur_it_token['tokens'] += [tokens[j]]
                
                it_text = ";".join(it_tokens)
                it_ids = []
                it_mask = []
                it_labels = []
                if it_text:
                    it_ids = tokenizer.encode(it_text)
                    it_mask = [1] * len(it_ids)
                    it_labels = [-100] * len(it_ids)

                start_it = len(prefix_ids + tokenized_inputs["input_ids"][i] + middle_ids)

                tokenized_inputs["input_ids"][i] = prefix_ids + tokenized_inputs["input_ids"][i] + middle_ids + it_ids + [tokenizer.eos_token_id]
                tokenized_inputs["attention_mask"][i] = prefix_mask + tokenized_inputs["attention_mask"][i] + middle_mask + it_mask + [1]
                tokenized_inputs["labels"][i] = prefix_labels + tokenized_inputs["labels"][i] + middle_labels + it_labels + [-100]

                batch_it_labels.append(
                    [-100]*start_it + tokenized_inputs["input_ids"][i][start_it:]
                )

                # debug.append(tokenizer.decode(tokenized_inputs["input_ids"][i]))

            tokenized_inputs["it_labels"] = batch_it_labels

            # tokenized_inputs["debug"] = debug
            
            return tokenized_inputs

        self.tokenized = dataset.map(tokenize_and_align_labels, batched=True)

        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        example = self.tokenized[idx]
        if self.max_length > len(example['input_ids']):
            input_ids = torch.tensor(example['input_ids'] + [self.pad_token_id] * (self.max_length - len(example['input_ids'])))
            attention_mask = torch.tensor(example['attention_mask'] + [0] * (self.max_length - len(example['attention_mask'])))
            labels = torch.tensor(example['labels'] + [-100] * (self.max_length - len(example['labels'])))
            it_labels = torch.tensor(example['it_labels'] + [-100] * (self.max_length - len(example['it_labels'])))
        else:
            input_ids = torch.tensor(example['input_ids'][:self.max_length - 1] + [example['input_ids'][-1]])
            attention_mask = torch.tensor(example['attention_mask'][:self.max_length])
            labels = torch.tensor(example['labels'][:self.max_length - 1] + [example['labels'][-1]])
            it_labels = torch.tensor(example['it_labels'][:self.max_length - 1] + [example['it_labels'][-1]])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'it_labels': it_labels
        }

