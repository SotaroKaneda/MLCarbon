import pandas as pd


class hardware_list:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.lifespan = 5 * 365

    def embodied_co2eq(self, execution_time, ground_truth):

        self.total_embodied_co2eq = 0.0

        for x in range(0, len(self.csv_file.index)):
            hardware_row = self.csv_file.iloc[x]
            name = hardware_row["hardware"]
            des = hardware_row["description"]
            unit = hardware_row["unit (cm2 or GB)"]
            CPA = hardware_row["CPA (kgCO2/cm2 or GB)"]
            number = hardware_row["num"]

            value = unit * CPA * number * execution_time / self.lifespan / 1000
            self.total_embodied_co2eq += value
            
        self.total_embodied_co2eq = self.total_embodied_co2eq / ground_truth - 1
