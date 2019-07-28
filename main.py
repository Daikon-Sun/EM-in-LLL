import os
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import logging
logger = logging.getLogger(__name__)


from settings import parse_args, model_classes
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, TimeFilter
from memory import Memory


def test_task(args, model, model_class, memory, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 6,
                                 num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    cur_loss, cur_acc, cur_n_inputs = 0, 0, 0
    for step, batch in enumerate(test_dataloader):
        model.eval()
        with torch.no_grad():
            n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
            outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
            if args.adapt_steps >= 1:
                q_input_ids, q_masks, q_labels = memory.query(input_ids, masks)
            else:
                loss, logits = outputs[:2]
                cur_n_inputs += n_inputs
                cur_loss += loss.item() * n_inputs
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                cur_acc += np.sum(preds == labels.detach().cpu().numpy())
    assert cur_n_inputs == len(test_dataset)
    logger.info("test loss: {:.3f} , test acc: {:.2f}".format(
        cur_loss / cur_n_inputs, cur_acc / cur_n_inputs))
    return cur_acc / cur_n_inputs


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

    updates_per_epoch = len(train_dataset)
    global_step = 0
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
        global_step += 1
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if global_step % args.logging_steps == 0:
            logger.info("progress: {:.2f} , global step: {} , lr: {:.2E} , avg loss: {:.3f}".format(
                (tot_n_inputs + 1) / updates_per_epoch, global_step,
                scheduler.get_lr()[0], tot_epoch_loss / tot_n_inputs))

            if args.debug:
                break

        if args.replay_interval >= 1 and (step + 1) % args.replay_interval == 0:
            input_ids, masks, labels = memory.sample(args.batch_size)
            loss = model(input_ids=input_ids, attention_mask=masks, labels=labels)[0]
            update_parameters(loss)

    assert tot_n_inputs == len(train_dataset)


def main():
    args = parse_args()

    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format,
                        filename=os.path.join(args.output_dir, 'log.txt'),
                        filemode='w', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

    logger.info("args: " + str(args))

    config_class, model_class, tokenizer_class = model_classes[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    config_save_path = os.path.join(args.output_dir, 'config')
    config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=config)
    model.to(args.device)
    memory = Memory(args)

    for task in args.tasks:
        train_dataset = TextClassificationDataset(task, "train", args, tokenizer)

        if args.valid_ratio > 0:
            valid_dataset = TextClassificationDataset(task, "valid", args, tokenizer)
        else:
            valid_dataset = None

        logger.info("Start training {}...".format(task))
        train_task(args, model, memory, train_dataset, valid_dataset)
        model_save_path = os.path.join(args.output_dir, 'model-' + task.split('/')[-1])
        torch.save(model.state_dict(), model_save_path)

    memory.build_tree()

    avg_acc = 0
    for task in args.tasks:
        test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
        task_acc = test_task(args, model, model_class, memory, test_dataset)
        avg_acc += task_acc / len(args.tasks)
    logger.info("Average acc: {:.2f}".format(avg_acc))


if __name__ == "__main__":
    main()
