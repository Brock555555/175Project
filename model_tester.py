from model import IdiomaticExpressionModel
from create_dataset import load_urban_dataset, prepare_dataset

if __name__ == "__main__":
    model = IdiomaticExpressionModel()
    ud_dataset = prepare_dataset(load_urban_dataset().sample(frac=1))

    model.train(ud_dataset)

    prompts = ["to make a mistake", "to die", "to date someone", "to drink alcohol"]
    for prompt in prompts:
        output = model.generate_idiom(f"Generate an idiom that means \"{prompt}\"")
        print(f"For definition {prompt}, the generated idiom was \"{output}\"")