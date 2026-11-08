<!-- Roll Number: 2025AG05790 | DM4ML Assignment 2 -->
# Data Validation Report

*Roll Number: 2025AG05790*

| Table | Check | Violations |
|---|---|---:|
| users | duplicate user_id | 5 |
| users | missing user_id | 0 |
| users | missing age (kept, imputed later) | 10 |
| users | age outside 5..100 | 0 |
| items | duplicate item_id | 0 |
| items | missing category | 0 |
| items | negative/non-numeric price (kept, imputed later) | 8 |
| interactions | missing user/item/rating | 754 |
| interactions | rating outside range | 757 |
| interactions | unparseable timestamp | 495 |
| interactions | timestamp out of range | 256 |
| interactions | exact duplicate rows | 688 |
| interactions | user_id not in users (RI) | 339 |
| interactions | item_id not in items (RI) | 731 |
| interactions | user-item pair repeated (keep latest) | 0 |

| Table | Rows in | Clean | Rejected |
|---|---:|---:|---:|
| users | 1005 | 1000 | 5 |
| items | 600 | 600 | 0 |
| interactions | 37664 | 33644 | 4020 |

_Roll Number: 2025AG05790_