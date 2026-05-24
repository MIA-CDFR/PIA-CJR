import torch


def process_outputs(outputs):

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    prediction = prediction.item()
    confidence = confidence.item() * 100

    return prediction, confidence
