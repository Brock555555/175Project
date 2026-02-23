from model import IdiomaticExpressionModel
from create_dataset import load_urban_dataset, prepare_dataset

if __name__ == "__main__":
    model = IdiomaticExpressionModel()
    ub_dataset = prepare_dataset(load_urban_dataset()  .sample(frac=0.2))

    model.train(ub_dataset)
    model.generate_idiom("Generate an idiom that means \"to die\"")