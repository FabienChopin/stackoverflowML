from fastapi import FastAPI, Path, Request
from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
import uvicorn
from typing import Annotated
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import pickle as pkl
import sklearn
import gc


app = FastAPI(
    title="SOFTags",
    description="Get relevant tags for your StackOverFlow posts !",
    version="2.0"
)
handler = Mangum(app)

def generate_html_response(post= "post", tags="tags"):
    if post == "post":
       html_content = f"""
       <html>
          <head>
             <title>SOFTags</title>
          </head>
          <body>
             <h1>Welcome to SOFTags 2.0 !</h1>
             Get ML based relevant {tags} for your StackOverFlow {post} <br> <br>
             Please provide a post in the URL as ---.aws/model/{post}
          </body>
       </html>
       """
    else:
       unpacked_tags = "<br>".join(tags[0])
       html_content = f"""
       <html>
          <head>
             <title>SOFTags Prediction</title>
          </head>
          <body>
             <h1>Your post :</h1>
             {post[0]}
             <h1>Tags proposition :</h1>
             {unpacked_tags}
          </body>
       </html>
       """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return generate_html_response()

@app.post("/model")
async def read_item(text:str="This is a lambda post for tags to be created"):
   embed = tf.saved_model.load("USE")
   text = [text]

   embedded_input = embed(text)
   del embed
   gc.collect()

   model = keras.models.load_model("best_model_USE.h5")
   BERTDF = pd.DataFrame(model.predict(embedded_input))
   del model
   gc.collect()

   top5BERT = BERTDF.apply(lambda x: pd.Series(x.nlargest(5).values), axis=1)
   top5BERT.columns=["1st","2nd","3rd","4th","5th"]
   top5BERT[["2nd","3rd","4th","5th"]] = top5BERT[["2nd","3rd","4th","5th"]].mask(top5BERT["1st"]<0.2,10)
   top5BERT[["2nd","3rd","4th","5th"]] = top5BERT[["2nd","3rd","4th","5th"]].mask(top5BERT["2nd"]<0.2,10)
   top5BERT[["3rd","4th","5th"]] = top5BERT[["3rd","4th","5th"]].mask(top5BERT["3rd"]<0.2,10)
   top5BERT[["4th","5th"]] = top5BERT[["4th","5th"]].mask(top5BERT["4th"]<0.2,10)
   top5BERT["5th"] = top5BERT["5th"].mask(top5BERT["5th"]<0.2,10)
   
   BERTDF[["1st","2nd","3rd","4th","5th"]] = top5BERT

   del top5BERT
   gc.collect()

   BERTDF["minimum"] = BERTDF[["1st","2nd","3rd","4th","5th"]].min(axis=1)

   BERTDF = BERTDF.iloc[:,0:-6] >= BERTDF['minimum'].values[:,None]
   BERTDF = BERTDF.replace(False,0).replace(True,1)

   with open('multiLabelBinarizer.pkl', 'rb') as f:
      mlb = pkl.load(f)
   result = mlb.inverse_transform(np.array(BERTDF))

   return generate_html_response(text,result)

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)
