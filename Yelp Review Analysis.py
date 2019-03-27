
# coding: utf-8

# # Yelp Review Analysis

# We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset/version/4).
# 
# Each observation in this dataset is a review of a particular business by a particular user.
# 
# The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business (Higher stars is better). In other words, it is the rating of the business by the person who wrote the review.
# 
# The "cool" column is the number of "cool" votes this review received from other Yelp users.
# 
# The "useful" and "funny" columns are similar to the "cool" column.
# 
# The goal of this project is to predict whether the customer will rate the business as GOOD, BAD or NEUTRAL.
# 
# We have information regarding the Stars that were allocated to a business by a user. Using this, we will create a new attribute that is CUSTOMER EXP which will categorize stars 1 as BAD experience, stars 2 & 3 as NEUTRAL and stars 4 & 5 as GOOD experience.
# 
# We will use Word clouds to obtain better infographic content of all the reviews.

# ## Importing libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud.wordcloud import WordCloud, STOPWORDS
from PIL import Image


# ## Loading the data and interpreting it

# In[3]:


yelp = pd.read_csv("C:\\Users\\Kush\\Desktop\\Applied Project\\Yelp\\yelp.csv")
yelp.head()


# In[4]:


yelp.info()


# In[5]:


yelp.describe()


# ## Creating features

# Here we create the Customer Experience column where we categorize the Stars given by customers to different business as GOOD, BAD and NEUTRAL.
# 
# Also, we create a new feature that is Text Length that gives the length of the reviews. This feature will give us an understanding of customer behavior and their experience.

# In[6]:


Cust = []
for i in yelp['stars']:
    if i==1:
        Cust.append('BAD')
    elif i==2 | i==3:
        Cust.append('NEUTRAL')
    else:
        Cust.append('GOOD')
        
yelp['Customer EXP'] = Cust
yelp['Customer EXP'].value_counts()
yelp['Text Length'] = yelp['text'].apply(lambda x: len(x.split()))
yelp.head()


# ## Exploratory data analysis

# In[7]:


a = sns.FacetGrid(data=yelp, col='Customer EXP', hue='Customer EXP', palette='plasma', size=5)
a.map(sns.distplot, "Text Length")
yelp.groupby('Customer EXP').mean()['Text Length']


# From the above histogram plots for all 3 categories of 'Customer EXP', we can conclude that people having GOOD experience wound up writing around 100 words while people having BAD or NEUTRAL experience ended up writing slightly more at around 200 words.

# In[8]:


plt.figure(figsize=(10,10))
sns.boxplot(x='stars', y='Text Length', data=yelp)


# In[9]:


plt.figure(figsize = (7,5))
sns.countplot('stars', data = yelp, palette="husl")


# In[10]:


plt.figure(figsize = (7,5))
sns.countplot('Customer EXP', data = yelp, palette="Oranges")


# Lets find the Correlation between COOL, USEFUL, FUNNY and TEXTLENGTH features from the data set when we group it by STARS.

# In[11]:


yelp.groupby('Customer EXP').mean().corr()


# In[12]:


plt.figure(figsize = (8,6))
sns.heatmap(yelp.groupby('Customer EXP').mean().corr(), cmap = "coolwarm", annot=True)


# ## Classification algorithms for our prediction

# ### Splitting the data

# In[13]:


x = yelp['text']
y = yelp['Customer EXP']
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# ## Engineering a Text cleaning function to remove the Punctuations and Stopwords from the data

# In[25]:


from nltk.corpus import stopwords
def text_clean(message):
    nopunc = [i for i in message if i not in string.punctuation]
    nn = "".join(nopunc)
    nn = nn.lower().split()
    nostop = [word for word in nn if word not in stopwords.words('english')]
    return(nostop)


# In[15]:


good = yelp[yelp['Customer EXP'] == 'GOOD']
bad = yelp[yelp['Customer EXP'] == 'BAD']
neu = yelp[yelp['Customer EXP'] == 'NEUTRAL']


# ## Cleaning the Review for BAD, NEUTRAL and GOOD by removing the stopwords and Punctuations

# In[16]:


good_bow = text_clean(good['text'])


# In[17]:


bad_bow = text_clean(bad['text'])


# In[18]:


neu_bow = text_clean(neu['text'])


# In[19]:


good_para = ' '.join(good_bow)
bad_para = ' '.join(bad_bow)
new_para = ' '.join(neu_bow)


# ### Word cloud to display the most common words in the Reviews where customer experience was GOOD

# In[20]:


stopwords = set(STOPWORDS)
stopwords.add('one')
stopwords.add('also')
mask_image = np.array(Image.open("C:\\Users\\Kush\\Desktop\\Applied Project\\Yelp\\thumbsup.jpg"))
wordcloud_good = WordCloud(colormap = "Paired",mask = mask_image, width = 300, height = 200, scale=2,max_words=1000, stopwords=stopwords).generate(good_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_good, interpolation="bilinear", cmap = plt.cm.autumn)
plt.axis('off')
plt.figure(figsize = (10,8))
plt.show()
wordcloud_good.to_file("good.png")


# ## Word cloud to display the most common words in the Reviews where customer experience was BAD

# In[21]:


stopwords = set(STOPWORDS)
stopwords.add('one')
stopwords.add('also')
stopwords.add('good')
mask_image1 = np.array(Image.open("C:\\Users\\Kush\\Desktop\\Applied Project\\Yelp\\thumbsdown.jpg"))
wordcloud_bad = WordCloud(colormap = 'tab10', mask = mask_image1, width = 300, height = 200, scale=2,max_words=10000, stopwords=stopwords).generate(bad_para)
plt.figure(figsize = (10,7))
plt.imshow(wordcloud_bad,interpolation='bilinear',cmap = plt.cm.autumn)
plt.axis('off')
plt.show()
wordcloud_bad.to_file('bad.png')


# ## Word cloud to display the most common words in the Reviews where customer experience was NEUTRAL

# In[22]:


stopwords = set(STOPWORDS)
wordcloud_neu = WordCloud(colormap = "plasma", width = 1100, height = 700, scale=2,max_words=500, stopwords=stopwords).generate(new_para)
plt.figure(figsize = (7,10))
plt.imshow(wordcloud_neu,cmap = plt.cm.autumn)
plt.axis('off')
plt.show()
wordcloud_neu.to_file('neu.png')


# ### Observations from the Word Cloud

# 1. The customers which reviewed a business to be *GOOD* used words such as *GOOD*, *TIME*, *FOOD*, *LOVE*, *PLACE*.
# 2. The businesses which where reviewed to be *NEUTRAL* had words such as *FOOD, REALLY, PLACE, ORDERED, WELL, NICE*
# 3. The customers which reviewed a business to be *BAD* i.e. Stars = 1 used words such as *TABLE, TIME, ORDER, SERVICE, EVEN, BETTER*
# 
# From these observations, we find that there are a lot of unique words in our reviews which can turn up as good classifiers for a business. These words can be used as independent variables to classfify the reviews and customer experience as GOOD, BAD or NEUTRAL.

# ## Let's use Naive Bayes Classifier and Support Vector machines to classify customer experience

# In[26]:


# Converting words into vector
from sklearn.feature_extraction.text import CountVectorizer
cv_transformer = CountVectorizer(analyzer = text_clean)


# In[ ]:


x = cv_transformer.fit_transform(x)

