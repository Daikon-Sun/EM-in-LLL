import os
import argparse
import shutil
import GPUtil
from multiprocessing import cpu_count
import torch
from pytorch_transformers import BertForSequenceClassification, BertTokenizer, BertConfig

model_classes = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'xlnet': (XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMForSequenceClassification, XLMTokenizer),
}
label_offsets = {
    'ag_news_csv': -1,
    'amazon_review_full_csv': 3,
    'dbpedia_csv': 8,
    'yahoo_answers_csv': 22,
    'yelp_review_full_csv': 3
}

def parse_args():
    parser = argparse.ArgumentParser("Lifelong Language Learning")

    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adapt_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fp16_test", action="store_true")
    parser.add_argument("--adapt_lr", type=float, default=2e-3)
    parser.add_argument("--adapt_lambda", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type selected in the list: " + ", ".join(model_classes.keys()))
    parser.add_argument("--n_labels", type=int, default=33)
    parser.add_argument("--n_neighbors", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=7600)
    parser.add_argument("--n_train", type=int, default=115000)
    parser.add_argument("--n_workers", type=int, default=cpu_count())
    parser.add_argument("--output_dir", type=str, default="output0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay_interval", type=int, default=100)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--tasks", nargs='+', default=["ag_news_csv"])
    parser.add_argument("--valid_ratio", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)

    args = parser.parse_args()

    if args.debug:
        args.n_train = 500
        args.n_test = 100
        args.output_dir = "output_debug"
        args.overwrite = True

    # args.n_gpu = torch.cuda.device_count()
    args.device_id = GPUtil.getFirstAvailable(maxLoad=0.05, maxMemory=0.05)[0]
    torch.cuda.set_device(args.device_id)
    memory_size = GPUtil.getGPUs()[args.device_id].memoryTotal
    if args.batch_size <= 0:
        args.batch_size = int(memory_size * 64)

    if os.path.exists(args.output_dir):
        if args.overwrite:
            choice = 'y'
        else:
            choice = input("Output directory ({}) exists! Remove? ".format(args.output_dir))
        if choice.lower()[0] == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            raise ValueError("Output directory exists!")
    else:
        os.makedirs(args.output_dir)
    return args

