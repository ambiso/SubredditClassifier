# Subreddit Classification

Will train a support vector machine on reddit posts to predict what subreddit a post could have come from.

## Setup

```
virtualenv -p python
. bin/activate
pip install -r requirements.txt
./sclf.py
```

## Example Output

```
./sclf.py 
Loading data from reddit.pickle
Got: 4250
Learning..
Learning data: 3401
Test data: 849
Evaluating
Accuracy:            0.846878680801
Normalized Accuracy: 0.846878680801
Micro Precision:     0.846878680801
Macro Precision:     0.87076061738
Micro Recall:        0.846878680801
Macro Recall:        0.791126215332
Input: How to compute if a program terminates 
'How to compute if a program terminates' => computerscience
Input: Why do spaghetti snap into three pieces?
'Why do spaghetti snap into three pieces?' => askscience
```
