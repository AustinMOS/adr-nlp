import matplotlib.cm as cm
import html
from IPython.display import display, HTML
import torch
import numpy as np
from transformers import pipeline

def value2rgba(x, cmap=cm.RdYlGn, alpha_mult=1.0):
    "Convert a value `x` from 0 to 1 (inclusive) to an RGBA tuple according to `cmap` times transparency `alpha_mult`."
    c = cmap(x)
    rgb = (np.array(c[:-1]) * 255).astype(int)
    a = c[-1] * alpha_mult
    return tuple(rgb.tolist() + [a])


def piece_prob_html(pieces, prob, sep=' ', **kwargs):
    html_code,spans = ['<span style="font-family: monospace;">'], []
    for p, a in zip(pieces, prob):
        p = html.escape(p)
        c = str(value2rgba(a, alpha_mult=0.5, **kwargs))
        spans.append(f'<span title="{a:.3f}" style="background-color: rgba{c};">{p}</span>')
    html_code.append(sep.join(spans))
    html_code.append('</span>')
    return ''.join(html_code)

def show_piece_attn(*args, **kwargs):
    from IPython.display import display, HTML
    display(HTML(piece_prob_html(*args, **kwargs)))

def split_text(x, max_length):
    length = len(x)
    if length > max_length:
        splits = length // max_length
        y = list()
        [y.append(torch.tensor([x[i : i + max_length]])) for i in range(0, splits*max_length, max_length)]
        if length % max_length > 0:
            y.append(torch.tensor([x[splits*max_length : length]]))
    else:
        y = list(torch.tensor([x]))
        
    return y

def nothing_ent(i, word):
    return {
    'entity': 'O',
    'score': 0,
    'index': i,
    'word': word,
    'start': 0,
    'end': 0
}

def generate_highlighted_text(model, tokenizer, text):
    ner_model = pipeline(
        'token-classification',
        model=model,
        tokenizer=tokenizer, 
        ignore_labels=None,
        device=0)
    result = ner_model(text)
    tokens = ner_model.tokenizer.tokenize(text)
    label_indeces = [i['index'] - 1 for i in result]

    entities = list()
    for i, word in enumerate(tokens):
        if i in label_indeces:
            entities.append(result[label_indeces.index(i)])
        else:
            entities.append(nothing_ent(i, word))
    entities = ner_model.group_entities(entities)
    spans = [e['word'] for e in entities]
    probs = [e['score'] for e in entities]
    return piece_prob_html(spans, probs, sep=' ')