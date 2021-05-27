import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

import vectorizer

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as logr
from sklearn.metrics import accuracy_score

def vectorize_capandhash(data): #how many caption words and how many hashtags
    listdata = []
    for rec in data:
        vecrec = vectorizer.convert_caption_to_vector(rec)
        listdata.append(vecrec)

    return np.asarray(listdata)

def vectorize_author(data): #is author verified or not
    listauth = []
    for rec in data:
        author = rec["authorMeta"]
        vecauth = vectorizer.convert_author_to_vector(author)
        listauth.append(vecauth)

    return np.asarray(listauth)

def vectorize_audio(data): #is music original or not
    listaudio = []
    for rec in data:
        audio = rec["musicMeta"]
        vecaud = vectorizer.convert_audio_to_vector(audio)
        listaudio.append(vecaud)

    return np.asarray(listaudio)

def vectorize_video(data): #time of video
    listvid = []
    for rec in data:
        video = rec["videoMeta"]
        vecvid = vectorizer.convert_video_metadata_to_vector(video)
        listvid.append(vecvid)

    return np.asarray(listvid)

def vectorize_commsandshares(data):
    listcomandshare = []
    for rec in data:
        veccs = vectorizer.convert_ground_truth_to_vector(rec).numpy()
        listcomandshare.append(veccs)

    return np.asarray(listcomandshare)

def lin_regression(X,Y):
    lin_model.fit(X,Y)
    lin_coef = lin_model.coef_[0]
    r2 = lin_model.score(X, Y)
    return r2
    
def multi_regression(X,Y):
    lin_model.fit(X,Y)
    lin_coef = lin_model.coef_[0]
    r2 = lin_model.score(X, Y)
    return r2
    # lin_model.predict
    #X[['a', 'b']]
    #for i in range(4):
    #print('Coefficient of '+ cols[i] + ' is ' + str(round(lin_model.coef_[i],2)))

def bayesian_regression(X,Y):
    bay_model.fit(X,Y)
    r2 = bay_model.score(X, Y)
    #logistic_model.predict

def graph_single_regression(X,Y, xtitle, ytitle):
    lin_model.fit(X,Y)
    yhat = lin_model.predict(X)
    plt.scatter(X, Y)
    plt.scatter(X, yhat)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title('Linear Relationship between '+xtitle+' and '+ ytitle)
    plt.legend(["Observed Values", "Predicted Values"])
    r2 = lin_model.score(X, Y)

def graph_bayesian_regression(X,Y,xtitle,ytitle):
    bay_model.fit(X,Y)
    yhat = bay_model.predict(X)
    plt.scatter(X, Y)
    plt.scatter(X, yhat)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title('Bayesian Relationship between '+xtitle+' and '+ ytitle)
    plt.legend(["Observed Values", "Predicted Values"])
    r2 = bay_model.score(X, Y)

lin_model = lr()
bay_model = linear_model.BayesianRidge(normalize=True)

file = open("trending.json", encoding= "utf8")
rawdata = json.load(file)
data = rawdata['collector']
file.close()

# CAPTION AND HASHTAGS
capsandhashs = vectorize_capandhash(data)
captions = capsandhashs[:, [0]]
hashtags = capsandhashs[:, [1]]

# AUTHOR
authors = vectorize_author(data)
authverified = authors[:, [2]]

# AUDIO
audios = vectorize_audio(data)
audioorg = audios[:, [2]]

# VIDEO
videos = vectorize_video(data)
duration = videos[:, [0]]

# SHARES AND COMMENTS
sharesandcomms = vectorize_commsandshares(data)
likes = sharesandcomms[:, [0]]
views = sharesandcomms[:, [1]]
shares = sharesandcomms[:, [2]]
comments = sharesandcomms[:, [3]]

#SINGLE LINEAR REGRESSION CASES
#lin_regression(shares, views)

#MULTI LINEAR REGRESSION
lelz1 = np.concatenate((captions,hashtags,authverified,audios,videos,authors),axis=1)

#multi_regression(lelz1, views)

#LOGISTIC REGRESSION



#GRAPH CASES
#graph_single_regression(shares,views, "Shares", "Views")
#plt.show()
#graph_single_regression(comments,views, "Comments", "Views")
#plt.show()
#graph_single_regression(shares,views, "Shares", "Views")
#graph_single_regression(comments,likes, "Comments", "Likes")
#graph_single_regression(hashtags,views, "Hashtags", "Views")
#plt.show()
#graph_single_regression(hashtags,likes, "Hashtags", "Likes")
#graph_single_regression(duration,views, "Duration", "Views")

#graph_single_regression(duration,likes, "Duration", "Likes")
#graph_single_regression(authverified,views, "Author_Verified", "Views")
#plt.show()

#graph_single_regression(authverified,likes, "Author_Verified", "Likes")
#graph_single_regression(audioorg,views, "Original_Audio", "Views")
#plt.show()

#graph_bayesian_regression(shares,views, "Shares", "Views")
#plt.show()
#graph_bayesian_regression(comments,views, "Comments", "Views")
#plt.show()

#graph_bayesian_regression(shares,views, "Shares", "Views")
#graph_bayesian_regression(comments,likes, "Comments", "Likes")
#plt.show()

#graph_bayesian_regression(hashtags,views, "Hashtags", "Views")
#plt.show()

#graph_bayesian_regression(hashtags,likes, "Hashtags", "Likes")
#plt.show()

#graph_bayesian_regression(duration,views, "Duration", "Views")
#graph_bayesian_regression(duration,likes, "Duration", "Likes")
#graph_bayesian_regression(authverified,views, "Author_Verified", "Views")
#graph_bayesian_regression(authverified,likes, "Author_Verified", "Likes")
#graph_bayesian_regression(audioorg,views, "Original_Audio", "Views")

singler2values = []
singledata = [captions, hashtags, duration, comments, authverified, audioorg, likes, shares]
singlenames = ['Captions', 'Hashtags', 'Duration', 'Comments', 'Verified Author', 'Original Audio', 'Likes', 'Shares'] 
for i in singledata:
    r2score = lin_regression(i, views)
    singler2values.append(r2score)

plt.bar(singlenames,singler2values)
plt.xlabel("Single Variable")
plt.ylabel("R^2 Values")
plt.title("R^2 Values for Views")
plt.show()

#MULTI LINEAR REGRESSION
multir2values = []
multidata = [(captions,hashtags), (duration, comments), (authverified, hashtags),
(shares, comments), (duration, audioorg), (likes, comments)]
multinames = ['Captions and Hashtags', 'Duration and Comments', 'Verified Author and Hashtags', 'Shares and Comments', 
'Duration and Original Audio', 'Likes and Comments']
for i in multidata:
    multiconcat = np.concatenate(i,axis=1)
    r2score = multi_regression(multiconcat , views)
    multir2values.append(r2score)

plt.bar(multinames,multir2values)
plt.xlabel("Multi Variables")
plt.ylabel("R^2 Values")
plt.title("R^2 Values for Views")
plt.show()
