

## Getting Started
 Use app.py to wrap the inference logic in a flask server to serve the model as a REST webservice:
    * Execute the command `python app.py` to run the flask app.
    * Go to the browser and hit the url `0.0.0.0:80` to get a message `Hello World!` displayed. **NOTE**: A permission error may be received at this point. In this case, change the port number to 5000 in `app.run()` command in `app.py`. 
    
  Post Method 
  
  http://192.168.1.2:80/predict
  Body 
  
  {
    "patientId":"678613" }
