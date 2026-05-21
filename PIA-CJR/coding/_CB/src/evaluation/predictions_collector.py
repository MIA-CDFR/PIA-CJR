import torch
import torch.nn.functional as F
from tqdm import tqdm

from .prediction_results import PredictionResults


class PredictionsCollector:

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def collect(self):

        self.model.eval()

        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():

            for images, labels in tqdm(
                self.dataloader,
                desc="Collecting predictions"
            ):

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                probs = F.softmax(outputs, dim=1)

                preds = torch.argmax(probs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        return PredictionResults(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )