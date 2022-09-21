# Serve model as a flask application

from flask import Flask, request, json

import torch
from torch.autograd import Variable
from loader import load_data

from model import MRNet

model = None
app = Flask(__name__)
THRESHOLD=0.50

def load_model():
    global model
    model = MRNet()
    model.load_state_dict(torch.load("/Users/bharat/Documents/work/ml/MRNetTrain/trainData/val0.0336_train0.0400_epoch9"))
    model.eval()

@app.route('/')
def home_endpoint():
    return '<center><h1>MRNet Knee <h1></center>'


@app.route('/predict', methods=['POST'])
def get_prediction():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = request.json
        print(data)
        id=data["patientId"]
        vol = load_data(id)
        pred = torch.sigmoid(model(vol))
        pred_npy = pred.data.cpu().numpy()[0][0]
        ACL_REQ=False
        if pred_npy > THRESHOLD:
            ACL_REQ=True
        dict={'ACL_REQ' :ACL_REQ , 'CF_LEVEL': pred_npy*100 }
        return str(dict)
    else:
        return 'Content-Type not supported!'


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=80)
