#!/bin/bash -i

ADAPT_STEPS=16
N_NEIGHBORS=16

if [ "$2" == "0" ]; then
    TASKS="yelp_review_full_csv ag_news_csv dbpedia_csv amazon_review_full_csv yahoo_answers_csv"
elif [ "$2" == "1" ]; then
    TASKS="dbpedia_csv yahoo_answers_csv ag_news_csv amazon_review_full_csv yelp_review_full_csv"
elif [ "$2" == "2" ]; then
    TASKS="yelp_review_full_csv yahoo_answers_csv amazon_review_full_csv dbpedia_csv ag_news_csv"
elif [ "$2" == "3" ]; then
    TASKS="ag_news_csv yelp_review_full_csv amazon_review_full_csv yahoo_answers_csv dbpedia_csv"
fi

CUDA_VISIBLE_DEVICES="$1" python3 train.py --tasks $TASKS --output_dir "output$2"

# CUDA_VISIBLE_DEVICES="$1" python3 test.py --tasks ag_news_csv --adapt_steps $ADAPT_STEPS --n_neighbors $N_NEIGHBORS --output_dir "output$2" --fp16_test --resume --logging_steps 1
# CUDA_VISIBLE_DEVICES="$1" python3 main.py --tasks $TASKS --adapt_steps $ADAPT_STEPS --n_neighbors $N_NEIGHBORS --output_dir "output$2" --fp16_test
# CUDA_VISIBLE_DEVICES="$1" python3 test.py --tasks $TASKS --adapt_steps $ADAPT_STEPS --n_neighbors $N_NEIGHBORS --output_dir "output$2" --fp16_test --resume --logging_steps 1
