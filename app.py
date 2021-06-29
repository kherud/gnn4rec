import pickle
import torch
from tokenizers import Tokenizer
from flask import Flask, request, send_from_directory, jsonify, render_template

from model import GNN, Dataloader

app = Flask(__name__)

N_AUTHOR_SEARCH = 16
N_TOP_RESULTS = 15
N_HOPS_NEIGHBORHOOD = 3
N_MIN_NODES = 1000

with open("resources/edges.pkl", "rb") as file:
    edges = pickle.load(file)
with open("resources/features.pkl", "rb") as file:
    features = pickle.load(file)
with open("resources/author_citations.pkl", "rb") as file:
    author_citations = pickle.load(file)
with open("resources/papers_by_author.pkl", "rb") as file:
    papers_by_author = pickle.load(file)
with open("resources/authors.pkl", "rb") as file:
    author2id = pickle.load(file)
    id2author = {v: k for k, v in author2id.items()}
with open("resources/keywords.pkl", "rb") as file:
    keyword2id = pickle.load(file)
    id2keyword = {v: k for k, v in keyword2id.items()}
tokenizer = Tokenizer.from_file("resources/tokenizer.json")

dataloader = Dataloader(edges=edges,
                        features=features,
                        tokenizer=tokenizer,
                        k_hops=N_HOPS_NEIGHBORHOOD,
                        n_min_nodes=N_MIN_NODES)
model = GNN(n_tokens=tokenizer.get_vocab_size(),
            n_keywords=len(keyword2id),
            n_authors=len(author2id))
weights = torch.load("resources/model.pt", map_location="cpu")
model.load_state_dict(weights)
del weights

def bad_request(message):
    response = jsonify({"error": message})
    response.status_code = 400
    return response


@app.route("/")
def hello_world():
    return send_from_directory("templates", "index.html")


@app.route("/author/<name>")
def author_page(name):
    if name not in author2id:
        author = None
    else:
        author = {
            "name": name,
            "keywords": [id2keyword[k] for k in features[author2id[name]]],
            "papers": list(sorted(papers_by_author[name], key=lambda x: x["year"], reverse=True)),
        }

    return render_template(
        "author.html",
        author=author
    )


@app.route("/authors", methods=["POST"])
def get_authors():
    args = request.get_json()
    if "search" not in args:
        return bad_request("Missing 'search' argument")

    search = args["search"].lower().strip()
    filtered_authors = filter(lambda author: search in author.lower(), author2id.keys())
    filtered_authors = sorted(filtered_authors, key=author_citations.get, reverse=True)
    return {"authors": list(filtered_authors)[:N_AUTHOR_SEARCH]}


@app.route("/predict", methods=["POST"])
def get_prediction():
    args = request.get_json()

    if "author" not in args:
        return bad_request("Please enter an author.")
    author = " ".join(x.capitalize() for x in args["author"].split())
    if author not in author2id:
        return bad_request("Invalid Author.")
    author_id = author2id[author]

    text = args["text"].strip()
    if "text" not in args or len(text) == 0:
        return bad_request("Please enter a text.")

    try:
        node_ids, adjacencies, text, keywords, keyword_mask = dataloader.get(author_id, text)
        predictions, keyword_attention = model.forward(node_ids, adjacencies, text, keywords, keyword_mask)
        # unique_keywords = torch.unique(keywords)
        # keyword_attention = keyword_attention[unique_keywords]

        top_adjacent, top_non_adjacent, top_keywords = [], [], []

        for argument in predictions.argsort(descending=True):
            # first node_id is reserved for whom the prediction is made for
            node_id = node_ids[1 + argument].item()
            author = id2author[node_id]
            if len(top_adjacent) < N_TOP_RESULTS and node_id in edges[author_id]:
                top_adjacent.append(author)
            if len(top_non_adjacent) < N_TOP_RESULTS and node_id not in edges[author_id]:
                top_non_adjacent.append(author)
            if len(top_adjacent) == N_TOP_RESULTS and len(top_non_adjacent) == N_TOP_RESULTS:
                break

        for argument in keyword_attention.argsort(descending=True)[:N_TOP_RESULTS]:
            keyword = id2keyword[argument.item()]
            keyword = " ".join(x.capitalize() for x in keyword.split())
            top_keywords.append(keyword)

        return {
            "num_nodes": node_ids.shape[0],
            "num_hops": N_HOPS_NEIGHBORHOOD + 2,  # 2 GNN layers
            "adjacent_authors": top_adjacent,
            "non_adjacent_authors": top_non_adjacent,
            "keywords": top_keywords,
        }
    except:
        return bad_request("Error during prediction.")
