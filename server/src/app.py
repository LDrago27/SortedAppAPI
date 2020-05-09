from flask import Flask, request
from flask_restplus import Api, Resource, fields
from PredictionModel import time_series_predict
from werkzeug.contrib.fixers import ProxyFix
import json

app = Flask(__name__)
api = Api(app)
ns = api.namespace('AnalysisScripts', description='Select Appropiate Endpoints')
app.wsgi_app = ProxyFix(app.wsgi_app)

TimeSeriesPredModel = ns.model("Model for Time Series Prediction",
                               {"data":
                                    fields.List(fields.Float, description="List of time series data as a List", required=True),
                                "tolerancelevel":
                                    fields.Integer(description="Tolerance Level", default=10)
                                })


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
        result = [ele.values.astype('float64')[0]  for ele in op]
        return result


if __name__ == '__main__':
        app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))

