import argparse
import os
import torch
import pickle
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import copy
import logging
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)

from settings import parse_args, model_classes
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, init_logging
from memory import Memory


def local_adapt(input_ids, labels, tmp_model, q_input_ids, q_masks, q_labels, args, org_params):

    q_input_ids = q_input_ids.to(args.device).detach()
    q_masks = q_masks.to(args.device).detach()
    q_labels = q_labels.to(args.device).detach()

    # optimizer = optim.RMSprop(tmp_model.parameters(), lr=args.adapt_lr)
    optimizer = optim.SGD(tmp_model.parameters(), lr=args.adapt_lr, momentum=0.9)
    # optimizer = AdamW(tmp_model.parameters(), lr=args.learning_rate, eps=1e-4)
    # optimizer = optim.Adam(tmp_model.parameters(), lr=args.adapt_lr)

    tmp_model.zero_grad()
    for step in range(args.adapt_steps):
        tmp_model.train()
        params = torch.cat([torch.reshape(param, [-1]) for param in tmp_model.parameters()], 0)
        loss = tmp_model(input_ids=q_input_ids, attention_mask=q_masks, labels=q_labels)[0] \
            + args.adapt_lambda * torch.sum((org_params - params)**2)
        loss.backward()
        optimizer.step()
        tmp_model.zero_grad()

    with torch.no_grad():
        tmp_model.eval()
        output = tmp_model(input_ids=input_ids, labels=labels)[:2]
        loss = output[0].item()
        logits = output[1].detach().cpu().numpy()
        torch.cuda.empty_cache()
        return loss, logits


def test_task(task_id, args, model, memory, test_dataset):

    if args.fp16_test:
        model = model.half()

    def update_metrics(loss, logits, cur_loss, cur_acc):
        preds = np.argmax(logits, axis=1)
        return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())

    cur_loss, cur_acc = 0, 0
    if args.adapt_steps >= 1:
        with torch.no_grad():
            org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

        q_input_ids = pickle.load(open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(task_id)), 'rb'))
        q_masks = pickle.load(open(os.path.join(args.output_dir, 'q_masks-{}'.format(task_id)), 'rb'))
        q_labels = pickle.load(open(os.path.join(args.output_dir, 'q_labels-{}'.format(task_id)), 'rb'))

        for i in range(len(test_dataset)):
            labels, input_ids = test_dataset[i]
            labels = torch.tensor(np.expand_dims(labels, 0), dtype=torch.long).to(args.device)
            input_ids = torch.tensor(np.expand_dims(input_ids, 0), dtype=torch.long).to(args.device)
            loss, logits = local_adapt(input_ids, labels, copy.deepcopy(model), q_input_ids[i], q_masks[i], q_labels[i], args, org_params)
            cur_loss, cur_acc = update_metrics(loss, logits, cur_loss, cur_acc)
            if (i+1) % args.logging_steps == 0:
                logging.info("Local adapted {}/{} examples, test loss: {:.3f} , test acc: {:.3f}".format(
                    i+1, len(test_dataset), cur_loss/(i+1), cur_acc/(i+1)))
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 6,
                                     num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
        tot_n_inputs = 0
        for step, batch in enumerate(test_dataloader):
            n_inputs, input_ids, masks, labels = prepare_inputs(batch, args.device)
            tot_n_inputs += n_inputs
            with torch.no_grad():
                model.eval()
                outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
                loss, logits = outputs[:2]
            cur_loss, cur_acc = update_metrics(loss.item()*n_inputs, logits, cur_loss, cur_acc)
            if (step+1) % args.logging_steps == 0:
                logging.info("Tested {}/{} examples , test loss: {:.3f} , test acc: {:.3f}".format(
                    tot_n_inputs, len(test_dataset), cur_loss/tot_n_inputs, cur_acc/tot_n_inputs))


    logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
        cur_loss / len(test_dataset), cur_acc / len(test_dataset)))
    return cur_acc / len(test_dataset)


def main():
    parser = argparse.ArgumentParser("Lifelong Language Learning")
    parser.add_argument("--output_dir", type=str, default="output0")
    output_dir = parser.parse_args().output_dir
    args = pickle.load(open(os.path.join(output_dir, 'args'), 'rb'))
    args.adapt_lambda = 0.1
    args.adapt_steps = 12
    args.adapt_lr = 1e-2
    args.fp16_test = True
    args.logging_steps = 1
    assert args.output_dir == output_dir
    init_logging(os.path.join(args.output_dir, 'log_test.txt'))
    logger.info("args: " + str(args))

    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    save_model_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(len(args.tasks)-1))
    model = model_class.from_pretrained(save_model_path, config=model_config).to(args.device)

    logger.info("Build tree...")
    memory = pickle.load(open(os.path.join(args.output_dir, 'memory-{}'.format(len(args.tasks)-1)), 'rb'))
    if args.adapt_steps >= 1:
        memory.build_tree()

    avg_acc = 0
    for task_id, task in enumerate(args.tasks):
        logger.info("Start testing {}...".format(task))
        test_dataset = pickle.load(open(os.path.join(args.output_dir, 'test_dataset-{}'.format(task_id)), 'rb'))
        task_acc = test_task(task_id, args, model, memory, test_dataset)
        avg_acc += task_acc / len(args.tasks)
    logger.info("Average acc: {:.3f}".format(avg_acc))


if __name__ == "__main__":
    main()
