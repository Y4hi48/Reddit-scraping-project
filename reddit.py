from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import praw
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

reddit=praw.Reddit(
    client_id="D5g3Wq6BmGcJCgfXOYyB-Q",
    client_secret="B-mZy2wiDVnU0O_kyzBE2mGIMIjEog",
    user_agent="Scraper 1.0 by /u/LaazyPixie"
)

headlines=set()
for subred in reddit.subreddit('politics').hot(limit=None):
  headlines.add(subred.title)
df=pd.DataFrame(headlines)

df.to_csv('headlines.csv',header=False,index=False,encoding='utf-8')

nltk.download('vader_lexicon')

sia=SIA()
results=[]
for line in headlines:
  pol_score=sia.polarity_scores(line)
  pol_score['headline']=line
  results.append(pol_score)
df=pd.DataFrame.from_records(results)


df['score']=0
df.loc[df['compound']<-0.2,'score']=-1
df.loc[df['compound']>0.2,'score']=1


DF=df[['headline','score']]


DF.to_csv('headline_score.csv',encoding='utf-8',index=False)

print(DF.score.value_counts())

print("Quelque positive headlines:")
pprint(list(DF[DF['score']==1].headline)[:5],width=200)

print("____________________________________________________________________________________________________________")

print("Quelque neutre headlines:")
pprint(list(DF[DF['score']==0].headline)[:5],width=200)

print("____________________________________________________________________________________________________________")

print("Quelque negative headlines:")
pprint(list(DF[DF['score']==-1].headline)[:5],width=200)

fig, ax = plt.subplots(figsize=(8, 8))
counts=DF.score.value_counts(normalize=True)*100
sns.barplot(x=counts.index,y=counts,ax=ax)
ax.set_xticklabels(['Negative','Neutre','Positive'])
ax.set_ylabel('Pourcentage')
ax.set_title('Répartition des scores')
plt.show()