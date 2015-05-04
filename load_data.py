import data_factory as df
from sklearn.preprocessing import OneHotEncoder

def load(binary_encode = False):
    data, label, desc, size = df.loadFirstCancerDataset()

    print binary_encode

    if binary_encode:
        enc = OneHotEncoder()
        enc.fit(data)
        encoded = enc.transform(data).toarray()
        return encoded
    else:
        return data