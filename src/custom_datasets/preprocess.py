import json
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

error_count = 0

def preprocess_ACE(
    input_dir: str,
    output_dir: str,
    model_name: str,
    max_length: int = 256,
    include_nan: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    # tokenizer.add_eos_token = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def read_jsonl_comprehension(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            result = []
            for line in file:
                row = json.loads(line)
                if len(row['event_mentions']) > 0:
                    del row['entity_mentions']
                    del row['relation_mentions']

                    event_mentions = []

                    for event_mention in row['event_mentions']:
                        event_mentions.append({
                            "event_type": event_mention['event_type'],
                            "trigger": event_mention['trigger']
                        })

                    result.append(
                        {
                            "sentence": row['sentence'],
                            "tokens": row['tokens'],
                            "event_mentions": event_mentions
                        }
                    )
            return result

    label2id = {'NAN': -100, 'O': 0}
    label_count = {'NAN': 0, 'O': 0}
        
    def convert_to_iob2(example):
        """
        Convert sentence and trigger to IOB2 format using transformer tokenizer
        
        Args:
            example: dict containing 'sentence' and 'event_mentions'
            tokenizer: transformer tokenizer
            max_length: max sequence length
        
        Returns:
            input_ids: tensor of token ids
            iob_labels: list of IOB2 labels with event type
        """
        
        # Tokenize the sentence
        tokenized = tokenizer(
            example['sentence'],
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Get the offset mapping (character positions for each token)
        offset_mapping = tokenized.offset_mapping[0].tolist()
        
        # Initialize IOB2 labels (O for all tokens initially)
        iob_labels = ['O'] * len(offset_mapping)
        if hasattr(tokenizer, 'add_bos_token') and tokenizer.add_bos_token:
            iob_labels = ['NAN'] + iob_labels[1:]
        if hasattr(tokenizer, 'add_eos_token') and tokenizer.add_eos_token:
            iob_labels = iob_labels[:-1] + ['NAN']
        
        debug = []

        for event_mention in example['event_mentions']:
            # Get trigger and event type information
            trigger_start = event_mention['trigger']['start']
            trigger_end = event_mention['trigger']['end']
            event_type = event_mention['event_type']
            original_text = event_mention['trigger']['text']
            
            # Find which transformer tokens correspond to the trigger
            trigger_token_indices = []
            
            # Get the character positions of the trigger in the original text
            original_tokens = example['tokens']
            trigger_start_char = 0
            for i in range(trigger_start):
                trigger_start_char += len(original_tokens[i]) + 1
            remaining_tokens = original_tokens[trigger_start:trigger_end]
            trigger_end_char = trigger_start_char + (len(' '.join(remaining_tokens)) + 1 if remaining_tokens else 0)

            
            # Find which transformer tokens overlap with the trigger
            for idx, (start, end) in enumerate(offset_mapping):
                # Skip special tokens ([CLS], [SEP], etc.)
                if start == end:
                    continue
                
                # Check if this token overlaps with the trigger
                if (start + 1 < trigger_end_char and end + 1 > trigger_start_char):
                    trigger_token_indices.append(idx)

            after_tagging = tokenizer.decode(tokenized['input_ids'][0][trigger_token_indices[0]:trigger_token_indices[-1]+1])
            debug.append({
                'original_text': original_text,
                'after_tagging': after_tagging
            })
            
            # debugging
            # if original_text.lower().strip() != after_tagging.lower().strip():
            #     global error_count
            #     error_count += 1
            #     print(f'Error count: {error_count}')
            #     print(f'Original text: {original_text}')
            #     print(f'After tagging: {after_tagging}')
            #     print(f'Example: {example["sentence"]}')
            #     print(f'Event mention: {event_mention}')
            #     print(f'Trigger token indices: {trigger_token_indices}')
            #     print(f'Offset mapping: {offset_mapping}')
            #     print(f'Tokenized: {tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])}')
            #     print(f'Debug: {debug}')
            #     print(f'Original tokens: {original_tokens}')
            #     print(f'Trigger start char: {trigger_start_char}')
            #     print(f'Trigger end char: {trigger_end_char}')
            #     print("="*50)
            #     if error_count == 3:
            #         exit()
            
            # Assign IOB2 labels with event type
            if trigger_token_indices:
                iob_labels[trigger_token_indices[0]] = f'B-{event_type}'
                for idx in trigger_token_indices[1:]:
                    iob_labels[idx] = f'I-{event_type}'
        
        for i in range(len(iob_labels)):
            if iob_labels[i] not in label2id:
                label2id[iob_labels[i]] = len(label2id) - 1
                label_count[iob_labels[i]] = 0
            
            label_count[iob_labels[i]] += 1
            iob_labels[i] = label2id[iob_labels[i]]


        input_ids = tokenized['input_ids'][0].tolist()
        attention_mask = tokenized['attention_mask'][0].tolist()

        # padding
        input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
        iob_labels = iob_labels + [label2id['NAN']] * (max_length - len(iob_labels))
        
        return {
            'debug': debug,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'iob_labels': iob_labels
        }
    
        """
        Convert sentence and trigger to IOB2 format using transformer tokenizer
        
        Args:
            example: dict containing 'sentence' and 'event_mentions'
        
        Returns:
            input_ids: tensor of token ids
            iob_labels: list of IOB2 labels with event type
        """
        
        # Tokenize the sentence
        tokenized = tokenizer(
            example['sentence'],
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Get the offset mapping (character positions for each token)
        offset_mapping = tokenized.offset_mapping[0].tolist()
        
        # Initialize IOB2 labels (O for all tokens initially)
        iob_labels = ['O'] * len(offset_mapping)
        if tokenizer.add_bos_token:
            iob_labels = ['NAN'] + iob_labels[1:]
        if tokenizer.add_eos_token:
            iob_labels = iob_labels[:-1] + ['NAN']
        
        
        # Get trigger and event type information
        trigger_start = example['event_mentions']['trigger']['start']
        trigger_end = example['event_mentions']['trigger']['end']
        event_type = example['event_mentions']['event_type']
        
        # Find which transformer tokens correspond to the trigger
        trigger_token_indices = []
        
        # Get the character positions of the trigger in the original text
        original_tokens = example['tokens']
        trigger_start_char = 0
        for i in range(trigger_start):
            trigger_start_char += len(original_tokens[i]) + 1
        trigger_end_char = trigger_start_char + len(original_tokens[trigger_start])
        
        # Find which transformer tokens overlap with the trigger
        for idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens ([CLS], [SEP], etc.)
            if start == end:
                continue
            
            # Check if this token overlaps with the trigger
            if (start <= trigger_end_char and end > trigger_start_char):
                trigger_token_indices.append(idx)
        
        # Assign IOB2 labels with event type
        if trigger_token_indices:
            iob_labels[trigger_token_indices[0]] = f'B-{event_type}'
            for idx in trigger_token_indices[1:]:
                iob_labels[idx] = f'I-{event_type}'
        
        return {
            'input_ids': tokenized['input_ids'][0],
            'attention_mask': tokenized['attention_mask'][0],
            'iob_labels': iob_labels
        }
    
    model_name = model_name.replace('/', '-')

    print(f'output_dir: {output_dir}')

    output_dir = os.path.join(output_dir, f'{model_name}' if not include_nan else f'include_nan-{model_name}')

    os.makedirs(output_dir, exist_ok=True)

    for split in os.listdir(input_dir):
        input_path = os.path.join(input_dir, split)

        print(f'Processing {split} split... at {input_path}')

        if "dev" in split:
            split = "dev.json"
        elif "test" in split:
            split = "test.json"
        elif "train" in split:
            split = "train.json"

        save_path = os.path.join(output_dir, split)
        data = read_jsonl_comprehension(input_path)
        with open(save_path, 'w') as f:
            for example in tqdm(data):
                processed = convert_to_iob2(example)
                f.write(json.dumps(processed) + '\n')

        print(f'Finished processing {split} split... at {save_path}')
    
    with open(os.path.join(output_dir, 'label2id.json'), 'w') as f:
        json.dump(label2id, f, indent=4)

    # calculate label distribution
    total_labels = sum(label_count.values())
    label_distribution = {k: v / total_labels for k, v in label_count.items()}
    statistic = {
        'label_count': label_count,
        'total_labels': total_labels,
        'label_distribution': label_distribution
    }

    with open(os.path.join(output_dir, 'statistic.json'), 'w') as f:
        json.dump(statistic, f, indent=4)

# def preprocess_conll2003(
#     output_dir: str,
#     model_name: str,
#     max_length: int = 256,
# )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', help='Dataset name')
    parser.add_argument('--input-dir', help='Input file dir')
    parser.add_argument('--output-dir', help='Output file dir')
    parser.add_argument('--model-name', help='Model name', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--max-length', help='Max length', type=int, default=256)
    parser.add_argument('--include-nan', help='Include NaN', action='store_true')
    args = parser.parse_args()

    dataset_name = args.dataset_name

    print(f'Preprocessing {dataset_name} dataset...')
    if dataset_name == 'ACE':
        preprocess_ACE(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_length=args.max_length,
            include_nan=args.include_nan
        )
    else:
        print(f'Unknown dataset: {dataset_name}')


if __name__ == '__main__':
    main()