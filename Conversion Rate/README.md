# How to Improve Conversion Rate

The data is about users hit into a website, the goal is to improve conversion rate.

## Data Description

The dataset contains 316200 instances and 6 features:

`country`: user country based on the IP address

`age` : user age. Self-reported at sign-up step

`new_user` : whether the user created the account during this session or had already an account and simply came back to the site

`source` : marketing channel source

`total_pages_visited`: number of total pages visited during the session. This can be seen as a proxy for time spent on site and engagement.

`converted`: this is our label. 1 means they converted within the session, 0 means they left without buying anything.

Conversion rate is calculated as `# conversions / totla sessions`.

## Data Analysis

Some exploratory data analysis is performed initially, every feature is analyzed and `groupby` by Python scripts. 

Later, I use graph to visualize the data and observe the patterns.

Advanced machine learning models (logistic regression & random forest) are trained and optimized. I also compare the two models' performance and use the best model to predict conversion rate.
