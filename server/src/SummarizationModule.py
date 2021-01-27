from gensim.summarization import summarize
from gensim.summarization import keywords
import gensim
import numpy as np
import pandas as pd


def summaryAndKeywords(text, model, word_count=80, keyword_count=20):
    summary = summarize(text, word_count=word_count)
    keyword = keywords(text, ratio=0.1, split=True, lemmatize=True, words=keyword_count)
    tag_pred = getRelevantTag(model,keyword)

    return [summary, keyword, tag_pred]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def getTagList():
    xls = pd.ExcelFile('Usertagging.xlsx')
    sheetname = list(xls.sheet_names)
    tag_list = []
    for name in sheetname:
        df = pd.read_excel(xls, name)
        df1 = df["Tag name"]
        df1 = df1.dropna()
        res = list(df1)
        tag_list.extend(res)
    return tag_list

def getRelevantTag(model,keyword):
    tag_list = getTagList();
    tag_score_list = []
    for word in keyword:
        for tag in tag_list:
            netscore = 0
            count = 0
            print(tag)
            for tagword in tag.split():
                if tagword in model and word in model:
                    score = cosine_sim(model[tagword], model[word])
                    netscore += score
                    count += 1
            if count:
                netscore = netscore / count
            tag_score_list.append([score, tag])
    tag_score_list.sort(reverse=True)
    return tag_score_list[:5]