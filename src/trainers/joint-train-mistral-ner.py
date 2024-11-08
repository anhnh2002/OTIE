import torch
from transformers import AutoTokenizer
from modelling.mistral import UnlockMistralForJointTokenClassificationCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from custom_datasets.dataset import JointConllDataset
from torch.utils.data import DataLoader

import evaluate

import logging
from dotenv import load_dotenv
import os
import argparse
from datetime import datetime
import sys
from tqdm import tqdm
import numpy as np


load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

def train(
    dataset_name: str,
    model_name: str,
    data_dir: str,
    output_dir: str,
    max_length: int,
    include_nan: bool,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    device: str,
    grad_accumulation_steps: int = 4,
    unlock: bool = False,
    log_step: int = 100,
    tc_loss_factor: float = 0.5,
    lm_loss_factor: float = 0.5
):
    # Log configuration
    logger.info('#' * 50)
    logger.info(f'Dataset: {dataset_name}')
    logger.info(f'Model: {model_name}')
    logger.info(f'Data dir: {data_dir}')
    logger.info(f'Output dir: {output_dir}')
    logger.info(f'Max length: {max_length}')
    logger.info(f'Include NaN: {include_nan}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Learning rate: {learning_rate}')
    logger.info(f'Number of epochs: {num_epochs}')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False


    # Load dataset
    train_dataset = JointConllDataset(dataset_name, 'train', tokenizer, max_length)
    dev_dataset = JointConllDataset(dataset_name, 'validation', tokenizer, max_length)
    test_dataset = JointConllDataset(dataset_name, 'test', tokenizer, max_length)
    num_labels = len(train_dataset.label2id)

    # Get dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # id2label
    id2label = train_dataset.id2label

    # Load model
    model = UnlockMistralForJointTokenClassificationCausalLM.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        device_map=device
    )

    # Load PEFT model
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.1
    )

    model = get_peft_model(model, peft_config)

    logger.info(model.print_trainable_parameters())

    # Get optimizer
    # trainable_params = []
    # for _, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         trainable_params.append(param)
    # optimizer = torch.optim.AdamW([{"params": trainable_params, "lr": learning_rate}])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)



    loss_fn = torch.nn.CrossEntropyLoss(
        # ignore_index=-100,
        # weight=class_weights
    )

    seqeval = evaluate.load("seqeval")

    label_list = list(id2label.values())

    # Get evaluation function
    def evaluate_token_classification(
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Evaluate token classification using sklearn metrics
        
        Args:
            true_labels: List of lists containing true labels
            pred_labels: List of lists containing predicted labels
        
        Returns:
            Dictionary containing metrics
        """

        predictions = logits.to(torch.float32).cpu().numpy()
        labels = labels.cpu().numpy()

        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        logger.info(f"true_predictions 0: {true_predictions[0]}")
        logger.info(f"true_labels 0: {true_labels[0]}")

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        logger.info("--" * 50)
        logger.info(f"Precision: {results['overall_precision']}")
        logger.info(f"Recall: {results['overall_recall']}")
        logger.info(f"F1: {results['overall_f1']}")
        logger.info(f"Accuracy: {results['overall_accuracy']}")
        logger.info("--" * 50)


        return results["overall_f1"]
        


    logger.info('#' * 50 + '\n\n')

    # Train model
    best_eval_f1 = 0
    test_f1_for_best_eval_f1 = 0
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        running_loss = 0
        for idx, batch in enumerate(train_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}

            labels=batch['labels']

            if idx == 0:
                optimizer.zero_grad()
            # outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            outputs = model(**batch, unlock_config=unlock)
            # loss = loss_fn(outputs.logits.view(-1, num_labels), labels.view(-1))
            loss = tc_loss_factor*outputs.tc_loss + lm_loss_factor*outputs.lm_loss
            loss.backward()
            running_loss += loss.item()
            if ((idx + 1) % grad_accumulation_steps == 0) or (idx == len(train_dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                if (idx + 1) % log_step == 0:
                    logger.info(f'Batch loss: {running_loss / grad_accumulation_steps}')
                running_loss = 0

            total_loss += loss.item()

        logger.info(f'Train loss: {total_loss / len(train_dataloader)}')

        # Evaluate
        logger.info('Evaluating on dev set...')
        model.eval()
        eval_loss = 0
        eval_f1 = 0
        entire_logits = []
        entire_labels = []
        for batch in dev_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            with torch.no_grad():
                # outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                outputs = model(**batch, unlock_config=unlock)
                logits = outputs.logits
                # loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))
                loss = tc_loss_factor*outputs.tc_loss + lm_loss_factor*outputs.lm_loss
                entire_logits.append(logits.cpu())
                entire_labels.append(batch['labels'].cpu())

                eval_loss += loss.item()
                # logger.info(f'Batch loss: {loss.item()}')

        logger.info(f'Dev loss: {eval_loss / len(dev_dataloader)}')
        eval_f1 = evaluate_token_classification(torch.cat(entire_logits, dim=0), torch.cat(entire_labels, dim=0))
        logger.info(f'Dev F1: {eval_f1}')


        if eval_f1 > best_eval_f1:
            best_eval_f1 = eval_f1
            logger.info(f'New best F1: {best_eval_f1}')
            model.save_pretrained(output_dir)

            # Evaluate on test set when there is improvement on dev set
            logger.info('Evaluating on test set...')
            test_loss = 0
            test_f1 = 0
            entire_logits = []
            entire_labels = []
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['labels']

                with torch.no_grad():
                    # outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    outputs = model(**batch, unlock_config=unlock)
                    logits = outputs.logits
                    # loss = loss_fn(outputs.logits.view(-1, num_labels), labels.view(-1))
                    loss = tc_loss_factor*outputs.tc_loss + lm_loss_factor*outputs.lm_loss
                    entire_logits.append(logits.cpu())
                    entire_labels.append(batch['labels'].cpu())

                    test_loss += loss.item()
                    # logger.info(f'Batch loss: {loss.item()}')


            logger.info(f'Test loss: {test_loss / len(test_dataloader)}')
            test_f1 = evaluate_token_classification(torch.cat(entire_logits, dim=0), torch.cat(entire_labels, dim=0))
            logger.info(f'Test F1: {test_f1}')

            test_f1_for_best_eval_f1 = test_f1

        if test_f1_for_best_eval_f1 >= 5:
            logger.info('Early stopping...')
            break

        # scheduler.step()

    logger.info('\n\n' + '**' * 50)
    logger.info('Training completed.')
    logger.info(f'Best F1: {best_eval_f1}')
    logger.info(f'Test F1 for best dev F1: {test_f1_for_best_eval_f1}')
    logger.info('**' * 50 + '\n\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', help='Dataset name')
    parser.add_argument('--data-dir', help='Input file dir')
    parser.add_argument('--output-dir', help='Output file dir')
    parser.add_argument('--model-name', help='Model name', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--max-length', help='Max length', type=int, default=256)
    parser.add_argument('--include-nan', help='Include NaN', action='store_true')
    parser.add_argument('--batch-size', help='Batch size', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('--num-epochs', help='Number of epochs', type=int, default=5)
    parser.add_argument('--grad-accumulation-steps', help='Gradient accumulation steps', type=int, default=4)
    parser.add_argument('--unlock', help='Unlock', action='store_true')
    parser.add_argument('--device', help='Device', default='cuda:0')
    parser.add_argument('--tc-loss-factor', help='Token classification loss factor', type=float, default=0.5)
    parser.add_argument('--lm-loss-factor', help='Language model loss factor', type=float, default=0.5)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    include_nan = args.include_nan
    unlock = True if args.unlock else False
    log_model_name = model_name.replace('/', '-')
    log_model_name = f"tc_{args.tc_loss_factor}-lm_{args.lm_loss_factor}-{log_model_name}"
    data_dir = os.path.join(args.data_dir, dataset_name, f'unlock-{log_model_name}' if unlock else log_model_name)
    output_dir = os.path.join(args.output_dir, dataset_name, f'joint-train-unlock-{log_model_name}' if unlock else f'joint-train-{log_model_name}', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    log_path = os.path.join(output_dir, "train.log")
    max_length = args.max_length

    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('train-base-ed')
    logger.setLevel(logging.INFO)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create handlers
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    

    train(
        dataset_name=dataset_name,
        model_name=model_name,
        data_dir=data_dir,
        output_dir=output_dir,
        max_length=max_length,
        include_nan=include_nan,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        grad_accumulation_steps=args.grad_accumulation_steps,
        unlock=unlock,
        device=args.device,
        tc_loss_factor=args.tc_loss_factor,
        lm_loss_factor=args.lm_loss_factor
    )

