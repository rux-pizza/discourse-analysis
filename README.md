# discourse-analysis

Analysis tools for Discourse posts using Python's scikit-learn.

Currently features a like predictor.

To use, use the following query to get data from the Discourse database and write the result to `data/posts.csv` in CSV format.

```sql
WITH ordered_post_actions AS
  (SELECT * FROM post_actions ORDER BY post_actions.created_at)
SELECT 
    times_from times_users_liked,
    users_from as users_liking,
    users2.username as posting_user,
    post_time,
    cooked
FROM (
    SELECT
        array_to_string(array_agg(post_actions.created_at),',') AS times_from,
        array_to_string(array_agg(u.username),',') AS users_from,
        posts.user_id AS user_to,
        posts.created_at AS post_time,
        posts.cooked,
        LENGTH(posts.cooked) as post_length
    FROM posts
    LEFT JOIN ordered_post_actions as post_actions
        ON post_id = posts.id AND post_action_type_id = 2 AND post_actions.deleted_at is null
    LEFT JOIN users as u
        ON u.id = post_actions.user_id
    INNER JOIN topics ON topics.id = posts.topic_id
    WHERE topics.archetype = 'regular'
    GROUP BY 
       posts.id
) AS c
LEFT JOIN users as users2 
    ON c.user_to = users2.id
ORDER BY post_time
```