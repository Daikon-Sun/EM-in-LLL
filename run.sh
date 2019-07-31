#!/bin/bash

# task order 1
python3 main.py \
  --tasks yelp_review_full_csv ag_news_csv dbpedia_csv amazon_review_full_csv yahoo_answers_csv \
  "$@"

# python3 main.py \
#   --tasks "datasets/yahoo_answers_csv" \
#   --batch_size 30 \
#   "$@"

# python3 main.py --tasks "datasets/ag_news_csv" --debug --adapt_steps 30 --output_dir output_test --batch_size 3
