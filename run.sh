#!/bin/bash

# task order 1
python3 main.py \
  --tasks "datasets/yelp_review_full_csv&datasets/ag_news_csv&datasets/dbpedia_csv&datasets/amazon_review_full_csv&datasets/yahoo_answers_csv" \
  "$@"
