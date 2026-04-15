import tweepy
from news_agent.config import settings

client = tweepy.Client(bearer_token=settings.twitter_bearer_token)
try:
    r = client.search_recent_tweets("AI", max_results=10)
    print("Success:", r.data)
except Exception as e:
    print(type(e).__name__, ":", e)
