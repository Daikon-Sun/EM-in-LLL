#!/bin/bash

# task order 1
python3 main.py \
  --tasks "datasets/yelp_review_full_csv&datasets/ag_news_csv&datasets/dbpedia_csv&datasets/amazon_review_full_csv&datasets/yahoo_answers_csv" \
  "$@"

# python3 main.py \
#   --tasks "datasets/yahoo_answers_csv" \
#   --batch_size 30 \
#   "$@"

# python3 main.py --tasks "datasets/ag_news_csv" --debug --adapt_steps 30 --output_dir output_test --batch_size 3
