import sys

def PredictionAccuracy(actual,predicted):
    assert len(actual)==len(predicted)
    hits=0.0
    for y,yhat in zip(actual,predicted):
        if y==yhat:
            hits=hits+1
    return 100*hits/len(actual)
