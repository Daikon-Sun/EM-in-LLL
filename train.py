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
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, init_logging, BatchSampler


def query_neighbors(task_id, args, memory, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 6,
                                 num_workers=args.n_workers, collate_fn=dynamic_collate_fn)


    q_input_ids, q_masks, q_labels = [], [], []
    for step, batch in enumerate(test_dataloader):
        n_inputs, input_ids, masks, labels = prepare_inputs(batch)
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

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
    #                               shuffle=not args.reproduce, collate_fn=dynamic_collate_fn)
    train_dataloader = DataLoader(train_dataset, num_workers=args.n_workers, collate_fn=dynamic_collate_fn,
                                  batch_sampler=BatchSampler(train_dataset, args.batch_size))
    # if valid_dataset:
    #     valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size * 6,
    #                                   num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_dataset)//4)

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
        n_inputs, input_ids, masks, labels = prepare_inputs(batch)
        memory.add(input_ids, masks, labels)
        loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
        update_parameters(loss)
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if (step+1) % args.logging_steps == 0:
            logger.info("progress: {:.2f} , global step: {} , lr: {:.2E} , avg loss: {:.3f}".format(
                tot_n_inputs/args.n_train, step+1, scheduler.get_lr()[0], tot_epoch_loss/tot_n_inputs))

        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            torch.cuda.empty_cache()
            del loss, input_ids, masks, labels
            input_ids, masks, labels = memory.sample(tot_n_inputs // (step + 1))
            loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
            update_parameters(loss)


    logger.info("Finsih training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
    del optimizer, optimizer_grouped_parameters
    assert tot_n_inputs == len(train_dataset) == args.n_train


def main():
    args = parse_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'args'), 'wb'))
    init_logging(os.path.join(args.output_dir, 'log_train.txt'))
    logger.info("args: " + str(args))

    logger.info("Initializing main {} model".format(args.model_name))
    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config).cuda()
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
        pickle.dump(memory, open(os.path.join(args.output_dir, 'memory-{}'.format(task_id)), 'wb'))


    if args.adapt_steps >= 1:
        memory.build_tree()
    del model

    for task_id, task in enumerate(args.tasks):
        test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
        pickle.dump(test_dataset, open(os.path.join(args.output_dir, 'test_dataset-{}'.format(task_id)), 'wb'))
        logger.info("Start querying {}...".format(task))
        query_neighbors(task_id, args, memory, test_dataset)


if __name__ == "__main__":
    main()
