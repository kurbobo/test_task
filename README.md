## test_task
make OCR model

In order to run everything in this repo you need to put test-task-data folder in the root path of this repo

data_exploration.ipynb is file, where I explored data and found out that there are some non digit examples (5), but 3 of them contains mistakes and such small amount of non digit capcha won't improve the recognition, so I deleted this 5 examples

captcha-with-pytorch.py is file with training and validating. The presicion on test data after 10 epochs with batchsize=64 was equal to 1. 

eval.png is photo with final evaluation result

model_only_numbers.pth is trained model
