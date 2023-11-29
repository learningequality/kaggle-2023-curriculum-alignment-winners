
# 1. Data explanation
- valid_samples_1k_first.csv , the topic_id is randomly selected which category != source, use it as a validation data

- learning-equality-curriculum-recommendations, this directory stores official datasets

# 2. Train Steps
- step1 code is prepare data train stage1 mode. Make the train_flag is True and False，you will get train data and validation data

- step2 code is train stage1 recall mode.

- step3 code is use stage1 recall mode to recall Top100 for all data 

- step4 code is prepare data train stage2 mode. Make the train_flag is True and False，you will get train data and validation data

- step5 code is train stage2 rank mode.

## solution ：
https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394838

## inference code ：
https://www.kaggle.com/code/xiamaozi11/lecr-submit-models-v2?scriptVersionId=121052259
