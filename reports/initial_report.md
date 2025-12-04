## First Draft of Final Project

### Initial Model Designs

In my initial EDA, I discovered that the Tenrec dataset can be used for a huge variety of recommendation tasks. For the project, I wanted to focus on classification (Click-through rate) and recommendation (Session-based recommendation). I started with session-based recommendation and immediately built a random baseline and evaluation framework. For SBR, HR@K and NDCG@K are great metrics, so I started with just those two.

The data is in the format: {user_id, item_id, click, follow, like, share, video_category, watching_times, gender, age}

### Features

For the SBR task, I found that features such as gender, age, and watching times would probably be less important in SBR. So, I focused on the item_id (video id), and the click/follow/like/share as metrics of engagement. I created a train/val/test split from the data by withholding the last item in each session for testing, the second to last for val, and the third to last for the target of training. Then, sessions were capped with a maximum of 30 videos in history, and a minimum of 10 for each model.

### Baseline Comparisons

First, the random baseline selects a random video and performs extremely poorly. There are roughly 130,000 unique videos, so this is expected to have an extremely low success rate. This model is pretty much useless other than a sign that it's difficult to have any HR@20 or NDCG@20.

Next, I calculated the most popular video by number of clicks and provided that recommendation for every session. This improved HR@20 and NDCG@20 significantly, so this is likely a better baseline than the random for my deep learning models. This model is extremely underfit, as it only recommends a single video to thousands of different users. However, this model essentially takes zero time to compute and could serve as a simple fallback for more complicated recommendation systems.

For the main model I plan to improve, I wanted to use a recurrent neural network to process the sequential nature of the data. I used an embedding layer for the video id, then added on the binary engagement metrics. This greatly improved recommendations, but I still have worries about data leakage. There is a high chance for overfitting, as the model can essentially memorize every sequence.

More specifically for the model, I used a embedding layer down to a vector of length 256, concatenating on the features, a two layer unidirectional LSTM with 512 hidden parameters, dropout=0.4, and finally a simple linear layer to all videos in the dataset. Sequences are selected with a minimum length of 10 videos, and a maximum of 30, so the most recent 30 videos are chosen. Shorter videos are left-padded so the last video is always in the same location. The model takes around 5 minutes to train three epochs on the smaller QK-Video dataset with ~30k users and ~130k videos on an A100 GPU.

### Evaluation 

As mentioned above, evaluation metrics fall in line to what we expect. Random is essentially useless, popular is better than nothing, and the deep learning approach has some real promise.

```bash
ubuntu@brev-qowxlk8zt:~/tenrec$ uv run eval.py --model random
random HR@20: 0.00%
random NDCG@20: 0.0000

ubuntu@brev-qowxlk8zt:~/tenrec$ uv run eval.py --model gru --checkpoint ./checkpoints/gru_last.pt
gru HR@20: 7.43%
gru NDCG@20: 0.0304

home@Mac tenrec % uv run eval.py --model popular
popular HR@20: 0.74%
popular NDCG@20: 0.0074
```

### Ideas for final report

I arbitrarly chose hyperparameters for the deep learning model and I also just created an architecture that would work in PyTorch. I would like to compare this to some off the shelf models tailor made for recommendations, like GRU4REC. Additionally, I want to create some better visualizations for the data, which might be difficult for SBR.

I also want to create another baseline like KNN to compare to my deep learning model. Right now any decent model will beat the most popular and random, so I want quantify the boosts of a deep learning model is worth the extra computation and complexity.

One error I may have made is with my data splitting. I train on every session in the dataset, and then just test on the withheld last interaction. However, I wonder if I should split all users into a 80/10/10 split and then try to recommend new sequences. After reading the Tenrec paper, I'm still a little confused on how they split the data for SBR, so I will confirm and adjust the split accordingly so I do not get artificially high evaluation metrics.

I would also like to explore CTR, so that would be something as a stretch goal for the final project.
