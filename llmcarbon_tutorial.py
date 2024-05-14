import argparse
import json
import pandas as pd

from model import LLM
from embodied import hardware_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=str, default="config_all.json")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            parser.set_defaults(**json.load(f))
        args = parser.parse_args()

    models_df = pd.read_csv(args.database)

    print("operational carbon footprint validation")

    for index_x in range(0, len(models_df.index)):
        model_row = models_df.iloc[index_x]

        model = LLM(
            model_row["name"],
            model_row["type"],
            model_row["parameter # (B)"],
            model_row["base model param. # (B)"],
            model_row["token # (B)"],
            model_row["CO2eq/KWh"],
            model_row["PUE"],
            model_row["computing device"],
            model_row["device TPD (W)"],
            model_row["avg. system power (W)"],
            model_row["peak TFLOPs/s"],
            model_row["achieved TFLOPs/s"],
            model_row["hardware efficiency"],
            model_row["device #"],
            model_row["total zettaFLOPs"],
            model_row["training days"],
            model_row["actual tCO2eq"],
        )
        model.training_co2eq()
        model.inference_co2eq()
        print(model.name + "\t" + str(model.train_delta))

    hardware_df = pd.read_csv(args.hardware)
    hard_list = hardware_list(hardware_df)

    print("\n\n")

    print("embodied carbon footprint validation")
    hard_list.embodied_co2eq(20.4, 0.66)
    print(hard_list.total_embodied_co2eq)


if __name__ == "__main__":
    main()
