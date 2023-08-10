import pandas as pd
import numpy as np
from model import Model
from training import Training
from datacenter import co2_oper
from datacenter import co2_emb




model_choice = int(input('Choose a Model\n1. Transformer(0.21 Billion Parameters)\n2. Evolved Transformer(0.13 Billion Parameters)\n3. Evolved Transformer NAS(0.064 Billion Parameters)\n4. T5(11 Billion Parameters)\n5. Meena\n6. Gshard\n7. Switch Transformer\n')) - 1
models_df = pd.read_csv('./data/models.csv')
model_row = models_df.iloc[model_choice]
model = Model(model_row['Number of Parameters (B)'], model_row['Tokens(trillions)'], model_row['Percent of model activated on every token'])
accelerator_choice = input('Choose acceleraton\n1. GPU\n2. TPU\n')

if accelerator_choice == '1':
    print('Calculating number of GPU and throughput using MegatronLM framework\n')
    gpu = int(input('Choose a GPU\n1. A100 80GB\n2. V100 32GB\n')) - 1
    gpus = ['A100', 'V100']
    accelerator = gpus[gpu]
    training = Training(model, accelerator, 0)
    node_size = int(input('Type the GPU node size, 8 is recommended\n'))
    training.predict_num_throughput(node_size)
    
        
if accelerator_choice == '2':
    tpu_version = int(input('Chose TPU version\n1. TPUv1\n2. TPUv2\n3. TPUv3\n4. TPUv4\n')) - 1
    tpus = ['TPUv1', 'TPUv2', 'TPUv3', 'TPUv4']
    accelerator = tpus[tpu_version]
    tpu_num = input('Type the number of TPU used for Training\nTo use standard values, leave blank\n')
    if not tpu_num:
        tpu_num = model_row['Number of Chips']
    training = Training(model, accelerator, tpu_num)
    training.calc_tpu_tflops()

# assuming 0.5 kg CO2 e/ KWh
impact = 500
# assuming PUE of 1.1
pue = 1.1
training.calc_train_time()
training.calc_energy()
datacenter_energy = training.total_energy * pue
operational_carbon = co2_oper(impact, datacenter_energy)

embodied_carbon = 0

if accelerator == 'V100':
    other_device_num = np.ceil(training.num / 8)
    # assuming one SSD, DRAM, CPU for every  8 GPU/TPU chip or one server stack
    devices = {"V100": training.num,
               "SSD": other_device_num,
               "CPU": other_device_num,
               "DRAM": other_device_num}
    for d in devices:
        embodied_carbon += co2_emb(d, devices[d], training.get_training_days())

print(f'MLCarbon Results for {model_row["Model"]}')
print(f'Number of Parameters: {model_row["Number of Parameters (B)"]} Billion')
print(f'Number of Tokens: {model_row["Tokens(trillions)"]} Trillion')
print(f'Training on {training.num} {accelerator}s\n')
print(f'LLM Training modeling Results')
tflops ="{:.2e}".format(model.total_tflops)
print(f'Total Computations: {tflops} TFLOPS')
print(f'Total Training days: {training.get_training_days()} days')
print(f'Total Training Energy: {datacenter_energy} MWh')
print(f'Total Operational Carbon Emissions: {operational_carbon} tons of CO2')
if embodied_carbon:
    print(f'Total Embodied Carbon: {embodied_carbon} tons of CO2\n')
