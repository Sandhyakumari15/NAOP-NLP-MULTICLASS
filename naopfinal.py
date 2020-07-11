

import pandas as pd
import re 
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, svm

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
np.random.seed(500)
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import wordcloud
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')

default_tokenizer=RegexpTokenizer(r"\w+")

from google.colab import drive
drive.mount('/content/drive')

Corpus=pd.read_csv('/content/drive/My Drive/data/NAOP_new.csv',encoding='latin1',usecols = ['Type','Posts'])

Corpus.head()

Corpus.shape

Corpus.columns

Corpus.isnull().sum()

"""# **EDA & Feature Engineering**"""



Corpus.groupby('Type').describe().T

Corpus['label'] = Corpus['Type'].map({'A': 0, 'B': 1,'C':2,'D':3})     # add 'label' column

sns.set_style('whitegrid')
sns.countplot(Corpus['Type'])
plt.title('Distribution of psychology')

# Make a new column to show the length of content messages
Corpus['length'] = Corpus['Posts'].apply(len)

import plotly.express as px
fig = px.violin(data_frame=Corpus, y="length", points="all", color="Type", 
                width=800, height=600)
fig.show()


colors = ['#ff9999','#66b3ff']
Corpus['Type'].value_counts().plot(kind = 'pie',colors = colors ,autopct = '%1.1f%%')

from collections import Counter
count1 = Counter(" ".join(Corpus[Corpus['Type']=='A']["Posts"]).split()).most_common(30)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in A", 1 : "count"})

count2 = Counter(" ".join(Corpus[Corpus['Type']=='B']["Posts"]).split()).most_common(30)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in B", 1 : "count"})

count3 = Counter(" ".join(Corpus[Corpus['Type']=='C']["Posts"]).split()).most_common(30)
df3 = pd.DataFrame.from_dict(count3)
df3 = df3.rename(columns={0: "words in C", 1 : "count"})

count4 = Counter(" ".join(Corpus[Corpus['Type']=='D']["Posts"]).split()).most_common(30)
df4 = pd.DataFrame.from_dict(count4)
df4 = df4.rename(columns={0: "words in D", 1 : "count"})

df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in A"]))
plt.xticks(y_pos, df1["words in A"])
plt.title('More frequent words in A')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in B"]))
plt.xticks(y_pos, df2["words in B"])
plt.title('More frequent words in B')
plt.xlabel('words')
plt.ylabel('number')
plt.show()



df3.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df3["words in C"]))
plt.xticks(y_pos, df3["words in C"])
plt.title('More frequent words in C')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


df3.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df4["words in D"]))
plt.xticks(y_pos, df4["words in D"])
plt.title('More frequent words in D')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


"""# **Data Preprocessing**"""

### We can see that the majority of frequent words in both classes are stop words such as 'to', 'a', 'or' and so on...
### so we remove these stopwords by putting them into 'WordCloud'

A = Corpus[Corpus["Type"] == "A"]
B = Corpus[Corpus["Type"]== "B"]
C = Corpus[Corpus["Type"]== "C"]
D = Corpus[Corpus["Type"]== "D"]
"""Wordcloud of all  class data"""



"""**Data Cleaning**"""

 
        
        


# Step - a : Remove blank rows if any.
Corpus['Posts'].dropna(inplace=True)
# Step - b : Change all the text to lower case. 
Corpus['Posts'] = [entry.lower() for entry in Corpus['Posts']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['Posts']= [word_tokenize(entry) for entry in Corpus['Posts']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['Posts']):
    # Declaring Empty List to store the words that follow the 1rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)
"""***WordCloud*** to check the unwanted meaningless words..."""
    
#wcloud = WordCloud(collocations=False,background_color='white').generate(' '.join(Corpus['text_final']))
#plt.imshow(wcloud)
#plt.axis('off')
#plt.show()

"""***WordCloud*** to check the unwanted meaningless words..."""

wcloud = WordCloud(collocations=False,background_color='white').generate(' '.join(Corpus['text_final']))
plt.imshow(wcloud)
plt.axis('off')
plt.show()
train = Corpus[['text_final', 'Type']]

train.head()
train['Type'] = train['Type'].map({'A': 0, 'B': 1,'C':2,'D':3})
X_train = train['text_final']
Y_train = train['Type']

# Vectorizer
# By using CountVectorizer function we can convert text document to matrix of word count.


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def vectorize_text(features, max_features):
    vectorizer = TfidfVectorizer( stop_words='english',
                            decode_error='strict',
                            analyzer='word',
                            ngram_range=(1, 2),
                            max_features=max_features,
                            max_df=0.5                    
                            )
    feature_vec = vectorizer.fit_transform(features)
    return feature_vec.toarray()

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

tf_idf_matrix
from sklearn import preprocessing

pickle.dump(count_vectorizer, open('tranform.pkl', 'wb'))



from imblearn.combine import SMOTETomek
oversample = SMOTETomek(random_state=42)
X_train_res, y_train_res = oversample.fit_sample(tf_idf_matrix, Y_train)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train_res, y_train_res,test_size=0.3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


print(sum(y_train==0))
print(sum(y_train==1))
print(sum(y_train==2))
print(sum(y_train==3))

#parameter tuning
#SVM = svm.SVC()
# defining parameter range 
#params = {'C': [1,10],  
              ##'gamma': [1.5,1, 0.1, 0.01, 0.001, 0.0001], 
             # 'kernel': ['linear','rbf']} 

#from sklearn.model_selection import  StratifiedKFold
##from sklearn.model_selection import RandomizedSearchCV
#folds = 5
#param_comb = 5
#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

#random_search = RandomizedSearchCV(SVM, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=skf.split(X_train_res, y_train_res), verbose=3, random_state=1001 )
#random_search.fit(X_train_res, y_train_res)
#print(random_search.cv_results_)
#print(random_search.best_estimator_)
#print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
#print(random_search.best_score_ * 2 - 1)
##print('\n Best hyperparameters:')
#print(random_search.best_params_)
# fitting the model for grid search 
SVM = svm.SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1.5, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

SVM.fit(X_train,y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_train)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_train)*100)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score,recall_score,accuracy_score, f1_score, precision_score
print(classification_report(predictions_SVM, y_train))
print(confusion_matrix(predictions_SVM, y_train))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_train, name='Actual'), pd.Series(predictions_SVM, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(predictions_SVM, y_train_res)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
predictions_SVM1 = SVM.predict(X_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM1, y_test)*100)
print(classification_report(predictions_SVM1, y_test))
print(confusion_matrix(predictions_SVM1, y_test))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(predictions_SVM1, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(predictions_SVM1, y_test)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
filename = 'nlp_model.pkl'
pickle.dump(SVM, open(filename, 'wb'))














