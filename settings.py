import os
import argparse
import shutil
from multiprocessing import cpu_count
import torch
from pytorch_transformers import BertForSequenceClassification, BertTokenizer, BertConfig

model_classes = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'xlnet': (XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMForSequenceClassification, XLMTokenizer),
}

def parse_args():
    parser = argparse.ArgumentParser("Lifelong Language Learning")

    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type selected in the list: " + ", ".join(model_classes.keys()))
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=7600)
    parser.add_argument("--num_train", type=int, default=115000)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--reproduce", action="store_true")
    # parser.add_argument("--tasks", type=str, default="datasets/dbpedia_csv")
    # parser.add_argument("--tasks", type=str, default="datasets/ag_news_csv&datasets/dbpedia_csv")
    parser.add_argument("--tasks", type=str, default="datasets/ag_news_csv")
    parser.add_argument("--valid_ratio", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    # parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        choice = input("Output directory ({}) exists! Remove? ".format(args.output_dir))
        if choice.lower()[0] == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            raise ValueError("Output directory exists!")
    else:
        os.makedirs(args.output_dir)
    args.tasks = args.tasks.split('&')
    args.n_gpu = torch.cuda.device_count()
    args.device = "cuda" if args.n_gpu > 0 else "cpu"
    return args

