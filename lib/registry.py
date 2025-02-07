import pickle


def load_model(local=True):
    if local:
        model = pickle.load(open(f"models/model.pkl", "rb"))
    else:
        model = pickle.load(open(f"/mnt/models/model.pkl", "rb"))
    return model
