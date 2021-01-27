import gensim
from flask import Flask, request
from flask_restplus import Api, Resource, fields
from PredictionModel import time_series_predict
from werkzeug.contrib.fixers import ProxyFix
import os

from SentimentModule import predictPositiveSentiProba
from sklearn.externals import joblib
# import joblib


from StreakModule import streakDataAnalyze

from SummarizationModule import summaryAndKeywords

from BaseRequestEnums import RequestType

app = Flask(__name__)
api = Api(app)
ns = api.namespace('AnalysisScripts', description='Select Appropriate Endpoints')
app.wsgi_app = ProxyFix(app.wsgi_app)
word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

TimeSeriesPredModel = ns.model("Model for Time Series Prediction",
                               {"data":
                                    fields.List(fields.Float, description="List of time series data as a List",
                                                required=True),
                                "tolerancelevel":
                                    fields.Integer(description="Tolerance Level", default=10)
                                })
SentiAnalyzerModel = ns.model("Model for Senti Analyzer",
                              {
                                  "text": fields.String(description="Enter the text whose Sentiment is to be Analyzed",
                                                        required=True)
                              })
StreakAnalyzerModel = ns.model("Model for Analyzing Streak Data",
                               {
                                   "data":
                                       fields.List(fields.String, description="List of Streak data as a List",
                                                   required=True),
                                   "analyzeEle":
                                       fields.String(description="Streak Data to be Analyzed for which Data element",
                                                     default="Yes")
                               })
TextSummazrizerModel = ns.model("Model for Text Summarizzation",
                                {
                                    "text": fields.String(description="Text that has to be Summarized",
                                                          default="Yes"),
                                    "Word_Count_Summary": fields.Integer(
                                        description="Enter Word Count in Summary Default is 80"),
                                    "keyword_Count": fields.Integer(
                                        description="Enter keyword word count default is 20 ")

                                })
BaseApiRequestModel=ns.model("Model for BASEAPICONTROLLER",
                             {
                                 "Request Type":fields.Integer(description="Request Type Refer to API contract",
                                                          default="Yes"),
                                 "UserID":fields.String(description="USERID"),
                                 "Periodicity":fields.Integer(description="Periodicity Enum Refer to API contract"),
                                 "QuestionID":fields.Integer(description="QuestionID"),
                                 "ActivityID":fields.Integer(description="ActivityID Valid only for Activity type questions"),
                                 "StartDate":fields.String(description="Start Date"),
                                 "EndDate":fields.String(description="End Date"),
                                 "NextN_Prediction":fields.Integer(description="Future Prediction of n days"),
                                 "Streak_Payload_for_hours":fields.String(description="JSON PAyload"),
                                 "Write_Payload:":fields.String(description="JSON PAyload")
                             }
                             )


SentiVectorizer = joblib.load('SentiVectorizer.pkl')
SentiClassifier = joblib.load('SentiClassifier.pkl')


@ns.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


@ns.route('/TimeSeriesPrediction')
class TimeSeriesPrediction(Resource):
    @ns.expect(TimeSeriesPredModel)
    def post(self):
        json_data = request.json
        op = time_series_predict(json_data['data'], json_data['tolerancelevel'])
        result = [ele.values.astype('float64')[0] for ele in op]
        return result


@ns.route('/SentiAnalyzer')
class SentiAnalyzer(Resource):
    @ns.expect(SentiAnalyzerModel)
    def post(self):
        json_data = request.json
        return predictPositiveSentiProba(json_data['text'], SentiVectorizer, SentiClassifier)


@ns.route('/StreakAnalyzer')
class StreakAnalyzer(Resource):
    @ns.expect(StreakAnalyzerModel)
    def post(self):
        '''

            :param arr: The array containing the streak information in form of strings
            :param analyze_ele: Which ele of streak is to be considered for analysis
            :return: length of longest streak, start of longest streak,current streak length
            in case start of longest streak -1 it means it does not exisist

        '''
        json_data = request.json
        op = streakDataAnalyze(json_data['data'], json_data['analyzeEle'])
        return op


@ns.route('/TextSummarizer')
class TextSummarizer(Resource):
    @ns.expect(TextSummazrizerModel)
    def post(self):
        json_data = request.json
        if "Word_Count_Summary" in json_data.keys() and json_data["Word_Count_Summary"]:
            word_count = json_data["Word_Count_Summary"]
        else:
            word_count = 80
        if "keyword_Count" in json_data.keys() and json_data["keyword_Count"]:
            keyword_count = json_data["keyword_Count"]
        else:
            keyword_count = 20
        op_dict = {}
        res = summaryAndKeywords(json_data["text"],word2vecModel, word_count, keyword_count)
        op_dict["summary"] = res[0]
        op_dict["keywords"] = res[1]
        op_dict["tags"] = res[2]
        return op_dict



if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))  # For Google Cloud Run Deployment
    app.run(debug=True)  # For Local PyCharm
