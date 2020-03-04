import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

from DistilBERT.model import DistilBERTModel
from GPT2.model import GPT2Model
from starlette.applications import Starlette
from starlette.responses import UJSONResponse
import uvicorn
import gpt_2_simple as gpt2
import tensorflow as tf
import os
import gc


# Initialise starlette app
app = Starlette(debug = True)

# Load pre-trained GPT2 Model
sess = gpt2.start_tf_sess(threads = 1)
gpt2_model = GPT2Model(tf_sess = sess)
gpt2_model.load_pretrained_model()

# Load pre-trained BERT Model
bert_model = DistilBERTModel()
bert_model.load_pretrained_model()

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}
generate_count = 0

@app.route('/', methods = ['GET', 'POST', 'HEAD'])
async def homepage(request):
    global generate_count
    global sess
    global gpt2_model
    global bert_model

    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    elif request.method == 'HEAD':
        return UJSONResponse({'subreddit': '', 'comments': ''},
                             headers = response_header)

    user_input = params.get('user_input', '****S')
    pred = bert_model.predict([user_input])
    comments = gpt2_model.generate_comments(user_input = user_input,
                                            top_k = 0,
                                            top_p = 0,
                                            bert_model_prediction = pred,
                                            length = int(params.get('length', 200)),
                                            temperature = float(params.get('temperature', 0.7)))
    # comments = list(comments)

    generate_count += 1
    if generate_count == 8:
        # Reload model to prevent Graph/Session from going OOM
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads = 1)
        gpt2_model = GPT2Model(tf_sess = sess)
        gpt2_model.load_pretrained_model()
        generate_count = 0

    gc.collect()
    return UJSONResponse({'subreddit': pred, 'comments': comments},
                         headers = response_header)

if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = int(os.environ.get('PORT', 8080)))



