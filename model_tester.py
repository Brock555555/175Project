from model import IdiomaticExpressionModel
from create_dataset import load_urban_dataset, prepare_dataset

if __name__ == "__main__":
    model = IdiomaticExpressionModel()
    ud_dataset = prepare_dataset(load_urban_dataset().sample(n=30000))

    model.train(ud_dataset, resume_from_checkpoint = False)

    # model.check_eos_token()
    # model.check_eos_function()

    prompts = ["to make a mistake", "to die", "to date someone", "to drink alcohol"]
    for prompt in prompts:
        output = model.generate_idiom(f"{prompt}")
        print(f"For definition {prompt}, the generated idiom was \"{output}\"")