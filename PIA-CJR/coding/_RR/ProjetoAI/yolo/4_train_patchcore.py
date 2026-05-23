from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder

def main():

    datamodule = Folder(
        name="solar",
        root="anomaly_dataset",
        normal_dir="train/good",
    )

    model = Patchcore()

    engine = Engine()

    engine.fit(
        model=model,
        datamodule=datamodule
    )

if __name__ == "__main__":
    main()