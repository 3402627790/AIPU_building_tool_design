from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
import onnx
import onnxruntime
import nlp
import numpy as np
import random
import torch
# The IMDb data set is a sentiment analysis data set (two classifications),
# test set each have 25000 samples (each sample is a movie review),
#  The number of samples of the positive/the negative class (ie positive/negative) is the same, 12500/12500.


def load_dataset(dataset: str, split: str = 'test'):

    data = nlp.load_dataset(dataset)
    txt, label = [], []
    for x in data[split]:
        txt.append(x['text'])
        label.append(x['label'])
    txt = np.array(txt)
    label = np.array(label)
    return txt, label


senti_txt, class_label = load_dataset('imdb', split='train')
print(senti_txt.shape, class_label.shape)
data_len = 5000
word_len = 385
inp_id = np.zeros((data_len, word_len), np.int32)
labels = np.ones(data_len)

txt_idx = random.sample(range(25000), data_len)
txt_content = senti_txt[txt_idx]
labels = class_label[txt_idx]
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
for i in range(data_len):
    text = txt_content[i]
    input_ids = (torch.tensor(tokenizer.encode(text, add_special_tokens=True)))[1:-1]  # Batch size
    tokens = input_ids.numpy()
    if tokens.size > word_len:
        tokens = tokens[0:word_len]
        # tokens[word_len - 1] = input_ids[-1]
    inp_id[i, 0:tokens.size] = tokens
np.save('dataset_'+str(data_len)+'.npy', inp_id)
np.save('labels_'+str(data_len)+'.npy', labels)

# verify it with onnx runtime
ort_session = onnxruntime.InferenceSession("t5-encoder.onnx")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)  # num_labels=2
count = 0
for i in range(data_len):
    ort_inputs = {ort_session.get_inputs()[0].name: np.expand_dims((inp_id[i]).astype(np.int64), 0)}
    ort_out = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_out)
    if pred == labels[i]:
        count = count + 1
print('acc is ', count/data_len)
