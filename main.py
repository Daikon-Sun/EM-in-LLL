import logging
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np

from settings import parse_args, model_classes
from utils import TextClassificationDataset, dynamic_collate_fn


def prepare_inputs(batch):
    num_inputs = len(batch[0])
    input_ids, masks, labels = tuple(b.to(args.device) for b in batch)
    return num_inputs, input_ids, masks, labels


def train_and_eval_task(task, args, model):
    config_class, model_class, tokenizer_class = model_classes[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    train_dataset = TextClassificationDataset(task, "train", args, tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=not args.reproduce,
                                  num_workers=args.num_workers,
                                  collate_fn=dynamic_collate_fn)

    valid_dataset = TextClassificationDataset(task, "valid", args, tokenizer)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size * 4,
                                  num_workers=args.num_workers,
                                  collate_fn=dynamic_collate_fn)

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
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                     t_total=total_train_step)

    logging.info("Start training...")

    train_per_epoch = len(train_dataset)
    valid_per_epoch = len(valid_dataset)
    avg_loss, total_num_inputs, global_step = 0, 0, 0
    model.zero_grad()

    for epoch in range(args.num_epochs):
        avg_epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            num_inputs, input_ids, masks, labels = prepare_inputs(batch)
            inputs = {'input_ids': input_ids, 'attention_mask': masks, 'labels': labels}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            total_num_inputs += num_inputs
            avg_loss += loss.item() * num_inputs
            avg_epoch_loss += loss.item() * num_inputs / train_per_epoch
            scheduler.step()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0:
                logging.info("progress: {:.2f}, global step: {}, lr: {:.2E}, avg loss: {:.3f}".format(
                    epoch + (step + 1) / train_per_epoch, global_step,
                    scheduler.get_lr()[0], avg_loss / total_num_inputs))
                avg_loss, total_num_inputs = 0, 0
        logging.info("epoch: {}, avg epoch loss: {:.3f}".format(epoch, avg_epoch_loss))

        eval_loss, eval_acc = 0, 0
        for step, batch in enumerate(valid_dataloader):
            model.eval()
            with torch.no_grad():
                num_inputs, input_ids, masks, labels = prepare_inputs(batch)
                inputs = {'input_ids': input_ids, 'attention_mask': masks, 'labels': labels}
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                eval_loss += loss.item() * num_inputs
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                eval_acc += np.sum(preds == labels.detach().cpu().numpy())
        logging.info("epoch: {}, eval loss: {:.3f}, eval acc: {}".format(
            epoch, eval_loss / valid_per_epoch, eval_acc / valid_per_epoch))


def run_task(task, args, model):

    if args.num_epochs > 0:
        train_and_eval_task(task, args, model)


if __name__ == "__main__":
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    model = None

    for task in args.tasks:
        run_task(task, args, model)

