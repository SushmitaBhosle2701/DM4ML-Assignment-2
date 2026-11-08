<!-- Roll Number: 2025AG05790 | DM4ML Assignment 2 -->
# Feature Catalog

*Roll Number: 2025AG05790*

All features are computed from the training split only and reused unchanged at prediction time via `get_user_features` / `get_item_features`.

| Entity | Feature | Type | Description | Definition |
|---|---|---|---|---|
| user | `user_avg_rating` | float | Mean rating given by the user (train) | mean(rating) per user_id |
| user | `user_activity_count` | int | Number of ratings given by the user | count(*) per user_id |
| user | `user_rating_std` | float | Rating spread (0 if one rating) | std(rating) per user_id |
| user | `days_since_last_interaction` | int | Recency vs. reference time (end of train) | ref_time - max(timestamp) |
| user | `preferred_category` | str | Category with most positive (>=4) ratings | argmax count(category | rating>=4) |
| user | `age` | int | User age (imputed with median) | users_clean.age |
| item | `item_popularity` | int | Number of ratings the item received | count(*) per item_id |
| item | `item_avg_rating` | float | Mean rating of the item | mean(rating) per item_id |
| item | `item_bayes_rating` | float | Shrunk mean rating (m=10 prior) | (n*mean + 10*global)/(n+10) |
| item | `category` | str | Primary category | items_clean.category |
| item | `category_onehot_*` | float | One-hot of primary/secondary category (used for diversity) | get_dummies |
| interaction | `user_item_category_affinity` | float | Share of user's ratings in item's category | n(user,cat)/n(user) |
| interaction | `rating_dev_from_user_mean` | float | rating - user_avg_rating | rating - user_avg_rating |

_Roll Number: 2025AG05790_
