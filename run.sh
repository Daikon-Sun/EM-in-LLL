#!/bin/bash -i

if [ "$1" == "0" ]; then
    TASKS="yelp_review_full_csv ag_news_csv dbpedia_csv amazon_review_full_csv yahoo_answers_csv"
elif [ "$1" == "1" ]; then
    TASKS="dbpedia_csv yahoo_answers_csv ag_news_csv amazon_review_full_csv yelp_review_full_csv"
elif [ "$1" == "2" ]; then
    TASKS="yelp_review_full_csv yahoo_answers_csv amazon_review_full_csv dbpedia_csv ag_news_csv"
elif [ "$1" == "3" ]; then
    TASKS="ag_news_csv yelp_review_full_csv amazon_review_full_csv yahoo_answers_csv dbpedia_csv"
fi

python3 train.py --tasks $TASKS --output_dir "output$2"
