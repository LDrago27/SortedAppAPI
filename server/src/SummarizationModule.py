from gensim.summarization import summarize
from gensim.summarization import keywords


def summaryAndKeywords(text, word_count=80, keyword_count=20):
    summary = summarize(text, word_count=word_count)
    keyword = keywords(text, ratio=0.1, split=True, lemmatize=True, words=keyword_count)
    return [summary, keyword]
