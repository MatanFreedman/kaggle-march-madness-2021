Summary
---------

The goal of this project was to predict winners and losers of the men's 2021 NCAA basketball tournament as part of a machine learning competition hosted on Kaggle.com. I used data of historical NCAA games provided by Kaggle and was encouraged to use other sources of publicly available data to gain a winning edge. The competition deliverable was a file with the probabilities of a win for every possible match up in the 2021 NCAA basketball tournament. 

![Bracket image](https://storage.googleapis.com/kaggle-competitions/kaggle/4862/media/bball-logo.png)

# Evaluation
Submissions were scored on the log loss:

<img src="https://render.githubusercontent.com/render/math?math=\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]">
                   
where

- <img src="https://render.githubusercontent.com/render/math?math=n"> is the number of games played
- <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted probability of team 1 beating team 2
- <img src="https://render.githubusercontent.com/render/math?math=y_i"> is 1 if team 1 wins, 0 if team 2 wins
- <img src="https://render.githubusercontent.com/render/math?math=log"> is the natural logarithm

The use of the logarithm provides extreme punishments for being both confident and wrong. In the worst possible case, a prediction that something is true when it is actually false will add an infinite amount to your error score. In order to prevent this, predictions are bounded away from the extremes by a small value.

# Data
I used the Kaggle provided detailed boxscore data combined with data I scraped from kenpom.com. My final dataset had the game results for every game in each tournament dating back to 2003. Each record in the dataset had 189 variables including averaged boxscore statistics (i.e., field goals made, 3-pointers made, offensive and defensive rebounds, assists, etc.), advanced stats (i.e., offensive efficiency, defensive efficiency, possession, assist ratio, etc.), 14-day win ratio prior to tournament, and kenpom statistics (i.e., adjusted efficiency margin, adjusted offensive efficiency, adjusted defensive efficiency, adjusted tempo, etc.). 

The final dataset had 2230 records, where each record represents one game with all the regular season statistics/variables calculated for T1_ and T2_. The outcome of each game was a "win" if T1_ Score was greater than T2_ Score. Each game had 2 rows where the second row had the T1_ and T2_ variables switched.

![Dataset image](https://github.com/MatanFreedman/kaggle-march-madness-2021/blob/master/reports/figures/dataset.PNG?raw=true)


# Modelling
From past competitions winning solutions I concluded that linear models were generally the best option because they were robust, fast, and easy to interpret. As this competition is a binary prediction a Logistic Regression model was an obvious choice, and generally did well in previous competitions (earning 2nd, 3rd, and 5th place in 2019). Other models that I tested included Linear Regression, Random Forests, Baysian Regression, and Ridge Regression. 

I used 'leave one group out' cross validation in an attempt accuratly score and compare my models. I trained each model using all seasons except for one, which I used to validate the model log loss score. This seemed to work well and was a recommended approach amoung many Kagglers.

![Training image](https://github.com/MatanFreedman/kaggle-march-madness-2021/blob/master/reports/figures/cv_training.PNG?raw=true)

# Final Model
The final model had a CV log_loss of 0.463, which was a pretty good Kaggle score. I was trying to be wary of overfitting, typically winning models had scores in the range of 0.45 - 0.55 and therefore I knew I was overfitting if I had scores below this range. 

My final model was an ensemble of 3 logistic regressions that used seperate datasets (one used boxscore and advanced stats, one used Kenny Pomeroy stats, and the other used a reduced feature set of all the features).

A histogram of the model output is shown below:

![Training image](https://github.com/MatanFreedman/kaggle-march-madness-2021/blob/master/reports/figures/win_probabilities.PNG?raw=true)

# Final Results
Unfortunately I believe by ensembling the models I overfit the data. Additionally, the Ken Pomeroy stats have data leakage in them, so that may have reduced the final score as well. My model had a log loss of 0.77456, which was much worse than my CV score when training. However, I also placed <img src="https://render.githubusercontent.com/render/math?math=1^st"> in my personal bracket competition with my friends, thus earning a cool $150 and bragging rights for the rest of the year.

Lessons learned:
- Use hypothesis testing to reduce features and avoid collinearity.
- Avoid using Kenny Pomeroy statistics because they include data leakage.
- Ensembling models may have caused overfitting.
