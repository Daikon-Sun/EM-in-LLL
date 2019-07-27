import os
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import logging
logger = logging.getLogger(__name__)

from settings import parse_args, model_classes
from utils import TextClassificationDataset, dynamic_collate_fn


def prepare_inputs(batch):
    num_inputs = len(batch[0])
    input_ids, masks, labels = tuple(b.to(args.device) for b in batch)
    return num_inputs, input_ids, masks, labels


def run_task(task, args, model):
    config_class, model_class, tokenizer_class = model_classes[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    train_dataset = TextClassificationDataset(task, "train", args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.reproduce,
                                  num_workers=args.num_workers, collate_fn=dynamic_collate_fn)

    if args.valid_ratio > 0:
        valid_dataset = TextClassificationDataset(task, "valid", args, tokenizer)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
                                      num_workers=args.num_workers, collate_fn=dynamic_collate_fn)

    test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 6,
                                 num_workers=args.num_workers, collate_fn=dynamic_collate_fn)

    config = config_class.from_pretrained(args.model_name, num_labels=train_dataset.num_labels, finetuning_task=task)

    if not model:
        model = model_class.from_pretrained(args.model_name, config=config)
        model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    total_train_step = len(train_dataloader) * args.num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_train_step)

    logger.info("Start training...")

    updates_per_epoch = len(train_dataset)
    global_step = 0
    model.zero_grad()

    for epoch in range(args.num_epochs):
        total_epoch_loss, total_num_inputs = 0, 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            num_inputs, input_ids, masks, labels = prepare_inputs(batch)
            inputs = {'input_ids': input_ids, 'attention_mask': masks, 'labels': labels}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            total_num_inputs += num_inputs
            total_epoch_loss += loss.item() * num_inputs
            scheduler.step()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0:
                logger.info("progress: {:.2f}, global step: {}, lr: {:.2E}, avg loss: {:.3f}".format(
                    epoch + (total_num_inputs + 1) / updates_per_epoch, global_step,
                    scheduler.get_lr()[0], total_epoch_loss / total_num_inputs))

        def run_evaluation(mode, dataloader):
            cur_loss, cur_acc, cur_num_inputs = 0, 0, 0
            for step, batch in enumerate(dataloader):
                model.eval()
                with torch.no_grad():
                    num_inputs, input_ids, masks, labels = prepare_inputs(batch)
                    inputs = {'input_ids': input_ids, 'attention_mask': masks, 'labels': labels}
                    outputs = model(**inputs)
                    loss, logits = outputs[:2]
                    cur_num_inputs += num_inputs
                    cur_loss += loss.item() * num_inputs
                    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                    cur_acc += np.sum(preds == labels.detach().cpu().numpy())
            logger.info("epoch: {}, {} loss: {:.3f}, {} acc: {}".format(
                epoch + 1, mode, cur_loss / cur_num_inputs, mode, cur_acc / cur_num_inputs))

        if valid_ratio > 0:
            run_evaluation("valid", valid_dataloader)
        run_evaluation("test", test_dataloader)


if __name__ == "__main__":
    args = parse_args()

    logging_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(format=logging_format,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=os.path.join(args.output_dir, 'log.txt'),
                        filemode='w', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger("").addHandler(console_handler)

    logger.info("args: " + str(args))

    model = None

    for task in args.tasks:
        run_task(task, args, model)

