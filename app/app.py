from typing import Union
from fastapi import FastAPI
from mangum import Mangum
from fastapi.responses import JSONResponse
import uvicorn
from typing import Union
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import pickle as pkl

app = FastAPI(
    title="SOFTags",
    description="Get relevant tags for your StackOverFlow posts !",
    version="1.0",
    docs_url='/docs',
    openapi_url='/openapi.json', # This line solved my issue, in my case it was a lambda function
    redoc_url=None
)
handler = Mangum(app)

#embed = tf.saved_model.load("app/universal-sentence-encoder_4")
embed=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
model = keras.models.load_model("app/best_model_USE.h5")
with open('app/multiLabelBinarizer.pkl', 'rb') as f:
  mlb = pkl.load(f)

@app.get("/")
def read_root():
   return {"Welcome to": "My first FastAPI deployment using Docker image"}

@app.get("/{text}")
def read_item(text: str):
   text = [text]
   embedded_input = embed(text)

   BERTDF = pd.DataFrame(model.predict(embedded_input))
   top5BERT = BERTDF.apply(lambda x: pd.Series(x.nlargest(5).values), axis=1)
   top5BERT.columns=["1st","2nd","3rd","4th","5th"]
   top5BERT[["2nd","3rd","4th","5th"]] = top5BERT[["2nd","3rd","4th","5th"]].mask(top5BERT["1st"]<0.2,10)
   top5BERT[["2nd","3rd","4th","5th"]] = top5BERT[["2nd","3rd","4th","5th"]].mask(top5BERT["2nd"]<0.2,10)
   top5BERT[["3rd","4th","5th"]] = top5BERT[["3rd","4th","5th"]].mask(top5BERT["3rd"]<0.2,10)
   top5BERT[["4th","5th"]] = top5BERT[["4th","5th"]].mask(top5BERT["4th"]<0.2,10)
   top5BERT["5th"] = top5BERT["5th"].mask(top5BERT["5th"]<0.2,10)
   
   BERTDF[["1st","2nd","3rd","4th","5th"]] = top5BERT
   BERTDF["minimum"] = BERTDF[["1st","2nd","3rd","4th","5th"]].min(axis=1)
   
   dfBERT = BERTDF.copy()
   dfBERT = dfBERT.iloc[:,0:-6] >= dfBERT['minimum'].values[:,None]

   predictionUSE = dfBERT.replace(False,0).replace(True,1)

   result = mlb.inverse_transform(np.array(predictionUSE))

   return JSONResponse({"result": result})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
   return JSONResponse({"item_id": item_id, "q": q})

if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=8080)
