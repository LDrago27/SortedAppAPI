from gensim.summarization import summarize
from gensim.summarization import keywords
import gensim
import numpy as np
import pandas as pd


def summaryAndKeywords(text, word_count=80, keyword_count=20):
    summary = summarize(text, word_count=word_count)
    keyword = keywords(text, ratio=0.1, split=True, lemmatize=True, words=keyword_count)

    return [summary, keyword]


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

