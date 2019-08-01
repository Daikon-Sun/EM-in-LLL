from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
import logging
import numpy as np
import os
import pickle
import torch
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from memory import Memory
from settings import parse_args, model_classes
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, TimeFilter


def query_neighbors(task_id, args, memory, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 8,
                                 num_workers=args.n_workers, collate_fn=dynamic_collate_fn)

    pickle.dump(test_dataloader, open(os.path.join(args.output_dir, 'dataloader-{}'.format(task_id)), 'wb'))

    q_input_ids, q_masks, q_labels = [], [], []
    for step, batch in enumerate(test_dataloader):
        n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
        with torch.no_grad():
            cur_q_input_ids, cur_q_masks, cur_q_labels = memory.query(input_ids, masks)
        q_input_ids.extend(cur_q_input_ids)
        q_masks.extend(cur_q_masks)
        q_labels.extend(cur_q_labels)
        if (step+1) % args.logging_steps == 0:
            logging.info("Queried {} examples".format(step+1))
    pickle.dump(q_input_ids, open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(task_id)), 'wb'))
    pickle.dump(q_masks, open(os.path.join(args.output_dir, 'q_masks-{}'.format(task_id)), 'wb'))
    pickle.dump(q_labels, open(os.path.join(args.output_dir, 'q_labels-{}'.format(task_id)), 'wb'))


def train_task(args, model, memory, train_dataset, valid_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.reproduce,
                                  num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    if valid_dataset:
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
                                      num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    tot_train_step = len(train_dataloader)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=tot_train_step)

    model.zero_grad()
    tot_epoch_loss, tot_n_inputs = 0, 0

    def update_parameters(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scheduler.step()
        optimizer.step()
        model.zero_grad()

    for step, batch in enumerate(train_dataloader):
        model.train()
        n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
        memory.add(input_ids, masks, labels)
        loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
        update_parameters(loss)
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if (step+1) % args.logging_steps == 0:
            logger.info("progress: {:.2f} , global step: {} , lr: {:.2E} , avg loss: {:.3f}".format(
                tot_n_inputs/len(train_dataset), step+1, scheduler.get_lr()[0], tot_epoch_loss/tot_n_inputs))

        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            input_ids, masks, labels = memory.sample(args.batch_size)
            loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
            update_parameters(loss)

        del loss, input_ids, masks, labels

    logger.info("Finsih training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
    del optimizer, optimizer_grouped_parameters
    assert tot_n_inputs == len(train_dataset)


def main():
    args = parse_args()


    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format,
                        filename=os.path.join(args.output_dir, 'log_train.txt'),
                        filemode='w', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

    logger.info("args: " + str(args))

    args.config_class, args.model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = args.config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    logger.info("Initializing main {} model".format(args.model_name))
    model = args.model_class.from_pretrained(args.model_name, config=model_config)
    model.to(args.device)
    memory = Memory(args)

    for task_id, task in enumerate(args.tasks):
        train_dataset = TextClassificationDataset(task, "train", args, tokenizer)

        if args.valid_ratio > 0:
            valid_dataset = TextClassificationDataset(task, "valid", args, tokenizer)
        else:
            valid_dataset = None

        logger.info("Start training {}...".format(task))
        train_task(args, model, memory, train_dataset, valid_dataset)
        model_save_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(task_id))
        torch.save(model.state_dict(), model_save_path)
        torch.cuda.empty_cache()
        pickle.dump(memory, open(os.path.join(args.output_dir, 'memory-{}'.format(task_id)), 'wb'))


    if args.adapt_steps >= 1:
        memory.build_tree()
    del model

    for task_id, task in enumerate(args.tasks):
        test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
        logger.info("Start querying {}...".format(task))
        query_neighbors(task_id, args, memory, test_dataset)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark=True
    main()
