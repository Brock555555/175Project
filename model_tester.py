from model import IdiomaticExpressionModel
from create_dataset import load_urban_dataset, prepare_dataset, load_idem_dataset
import json

def load_definitions():
    with open("idiom_defs.json", "r") as f:
        data = json.load(f)

    definitions = [item["definition"] for item in data["idioms"]]

    return definitions

if __name__ == "__main__":
    model = IdiomaticExpressionModel()
    # ud_dataset = prepare_dataset(load_urban_dataset().sample(n=30000))
    idem_dataset = load_idem_dataset()
    train_dataset = prepare_dataset(idem_dataset)

    model.train(train_dataset, resume_from_checkpoint = False)

    # model.check_eos_token()
    # model.check_eos_function()

    prompts = load_definitions()
    outputs = []
    for prompt in prompts:
        output = model.generate_idiom(f"{prompt}")
        outputs.append(output)
        print(f"For definition {prompt}, the generated idiom was \"{output}\"")

    with open("fine_tuned_generated_idioms.json", "w") as f:
        json.dump(outputs, f)

