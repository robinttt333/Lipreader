import config
from train import Trainer
from validation import Validation
from models.lipReader import Lipreader
from tqdm import tqdm
from datetime import datetime, timedelta
from savingAndLoading import saveModel, loadModel
if __name__ == "__main__":
    lipreader = Lipreader()
    trainer = Trainer(lipreader)
    validator = Validation(lipreader)

    print("Started training at", datetime.now())
    with tqdm(total=config.training["epochs"], desc="Epochs", position=0) as t:
        for epoch in range(config.training["completedEpochs"], config.training["epochs"]):
            trainer.train(epoch)
            validator.validate()
            t.update()
            saveModel(lipreader, epoch+1)
    loadModel(1)
