from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import  sentiment_module as s


# consumer key, consumer secret, access token, access secret.
ckey = "VlxElvtYpsxA1mLBYYHj5sNhY"
csecret = "o4AhWVbZdE8MQrF7e7DuzRcruvsZKM92G8jXPsZlnKuhnJtUcL"
atoken = "768869757542141953-9QgjNxQSReBB3M07guPoaEjzShuYqfk"
asecret = "qNRujtjAgtcMGqLfPNiql0D3Ig8N9Xw43s8gdGFFOdom7"


class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)

            tweet = all_data["text"]
            (sentiment_val, confidence) = s.sentiment(tweet)
            print(tweet, sentiment_val, confidence)

            if confidence >= 60:
                output = open("twitter-output.txt", "a")
                output.write(sentiment_val)
                output.write('\n')
                output.close()

            print(tweet)

            return True
        except:
            return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["obama"])
