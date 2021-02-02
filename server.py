from flask import Flask,request,jsonify
from GarbageDetectionTest import DetectGarbage
from GarbageClassificationTest import ClassifyGarbage
UPLOAD_FOLDER = 'Upload_image/'


app = Flask("__GarbageDetector__")
object = DetectGarbage()
obejctclass = ClassifyGarbage()

@app.route("/predict", methods = ["GET","POST"])
def predict():
   query_parameters = request.args
   path = query_parameters.get('path')
   result = object.classify(path)
   if result == "Garbage":
      type = obejctclass.classify(path)
   else:
      type = "NULL"
   ret = {"result":result,"type":type}
   return jsonify(ret)

app.run(debug=True, port=5965)
