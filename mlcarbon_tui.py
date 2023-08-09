import pandas as pd
from model import Model
from training import Training


model_choice = int(input('Choose a Model\n1. Transformer(0.21 Billion Parameters)\n2. Evolved Transformer(0.13 Billion Parameters)\n3. Evolved Transformer NAS(0.064 Billion Parameters)\n4. T5(11 Billion Parameters)\n5. Meena\n6. Gshard\n7. Switch Transformer\n')) - 1
models_df = pd.read_csv('./data/models.csv')
model_row = models_df.iloc[model_choice]
model = Model(model_row['Number of Parameters (B)'], model_row['Tokens(trillions)'], model_row['Percent of model activated on every token'])
accelerator_choice = input('Choose acceleraton\n1. GPU\n2. TPU\n')

if accelerator_choice == '1':
    print('Calculating number of GPU and throughput using MegatronLM framework\n')
    gpu = input('Choose a GPU\n1. A100 80GB\n 2. V100 32GB\n')
    training = Training(model, gpu, 0)
    node_size = input('Type the GPU node size, 8 is recommended')
    training.predict_num_throughput(node_size)
    
        
if accelerator_choice == '2':
    tpu_version = int(input('Chose TPU version\n1. TPUv1\n2. TPUv2\n3. TPUv3\n4. TPUv4\n')) - 1
    tpus = ['TPUv1, TPUv2, TPUv3, TPUv4']
    tpu = tpus[tpu_version]
    tpu_num = input('Type the number of TPU used for Training\nTo use standard values, leave blank\n')
    if not tpu_num:
        tpu_num = model_row['Number of Chips']
    training = Training(model, tpu, tpu_num)
    training.calc_tpu_tflops()

training.calc_train_time()
training.calc_energy()