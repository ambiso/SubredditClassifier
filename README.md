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
Receiving data
100%|██████████████████████████████████████████████████████████████████████| 5/5 [01:16<00:00, 14.27s/it]
Saving to reddit.pickle
Got: 4177
Learning..
Learning data: 3342
Test data: 835
Evaluating
Accuracy:            0.858682634731
Normalized Accuracy: 0.858682634731
Micro Precision:     0.858682634731
Macro Precision:     0.893765739361
Micro Recall:        0.858682634731
Macro Recall:        0.762565951694
Input: Why do spaghetti snap into three pieces?
'Why do spaghetti snap into three pieces?' => askscience
Input: Is there an algorithm to compute if a program halts?
'Is there an algorithm to compute if a program halts?' => computerscience
```
