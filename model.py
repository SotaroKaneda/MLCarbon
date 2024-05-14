import json
import csv
import numpy as np


class LLM:
    def __init__(
        self,
        name,
        type,
        parameters_b,
        base_param_b,
        tokens_t,
        co2eqkwh,
        pue,
        device_t,
        tpd,
        system_power,
        pTFLOPs,
        achievedTFLOPs,
        hardware_efficiency,
        device_num,
        total_FLOPs_z,
        training_days,
        actual_value,
    ):
        self.name = name
        self.type = type
        self.parameters_b = parameters_b
        self.base_param_b = base_param_b
        self.tokens_t = tokens_t
        self.co2eqkwh = co2eqkwh
        self.pue = pue
        self.device_t = device_t
        self.tpd = tpd
        self.system_power = system_power
        self.pTFLOPs = pTFLOPs
        self.achievedTFLOPs = achievedTFLOPs
        self.hardware_efficiency = hardware_efficiency
        self.device_num = device_num
        self.total_tflops = total_FLOPs_z
        self.training_days = training_days
        self.actual_value = actual_value

    # self.total_tflops = 6 * self.activated_parameters_b * tokens_t * 1e9

    def print(self):
        print("name\t\t\t" + self.name)
        print("type\t\t\t" + self.type)
        print("parameters_b\t\t\t" + str(self.parameters_b))
        print("base_param_b\t\t\t" + str(self.base_param_b))
        print("tokens_t\t\t\t" + str(self.tokens_t))
        print("co2eqkwh\t\t\t" + str(self.co2eqkwh))
        print("pue\t\t\t" + str(self.pue))
        print("device_t\t\t\t" + self.device_t)
        print("tpd\t\t\t" + str(self.tpd))
        print("system_power\t\t\t" + str(self.system_power))
        print("pTFLOPs\t\t\t" + str(self.pTFLOPs))
        print("achievedTFLOPs\t\t\t" + str(self.achievedTFLOPs))
        print("hardware_efficiency\t\t\t" + str(self.hardware_efficiency))
        print("device_num\t\t\t" + str(self.device_num))
        print("total_tflops\t\t\t" + str(self.total_tflops))
        print("training_days\t\t\t" + str(self.training_days))
        print("actual_value\t\t\t" + str(self.actual_value))

    def training_co2eq(self):
        if self.type == "MoE":
            total_FLOPs_all = (
                6
                * self.base_param_b
                * 1000
                * 1000
                * 1000
                * self.tokens_t
                * 1000
                * 1000
                * 1000
            )
        else:
            total_FLOPs_all = (
                6
                * self.parameters_b
                * 1000
                * 1000
                * 1000
                * self.tokens_t
                * 1000
                * 1000
                * 1000
            )

        self.total_FLOPs = total_FLOPs_all

        TFLOPSperSecond = self.pTFLOPs * self.hardware_efficiency
        training_day_num = (
            self.total_FLOPs
            / self.device_num
            / TFLOPSperSecond
            / 1000000000000
            / 3600
            / 24
        )

        self.actual_training_days = training_day_num

        energyMWh = (
            self.system_power
            * self.device_num
            * self.actual_training_days
            / 1000000
            * 24
            * self.pue
        )

        self.energy_all = energyMWh
        self.predicted_co2eq_train = self.energy_all * self.co2eqkwh
        self.train_delta = self.predicted_co2eq_train / self.actual_value - 1

        # print(self.predicted_co2eq_train)
        #print(self.train_delta)

    def inference_co2eq(self):
        if self.type == "MoE":
            total_FLOPs_all = (
                2
                * self.base_param_b
                * 1000
                * 1000
                * 1000
                * self.tokens_t
                * 1000
                * 1000
                * 1000
            )
        else:
            total_FLOPs_all = (
                2
                * self.parameters_b
                * 1000
                * 1000
                * 1000
                * self.tokens_t
                * 1000
                * 1000
                * 1000
            )

        self.total_infer_FLOPs = total_FLOPs_all

        TFLOPSperSecond = self.pTFLOPs * self.hardware_efficiency
        inference_time = (
            self.total_infer_FLOPs
            / self.device_num
            / TFLOPSperSecond
            / 1000000000000
            / 3600
            / 24
        )

        self.actual_inference_time = inference_time

        energyMWh = (
            self.system_power
            * self.device_num
            * self.actual_inference_time
            / 1000000
            * 24
            * self.pue
        )

        self.energy_infer_all = energyMWh
        self.predicted_co2eq_infer = self.energy_infer_all * self.co2eqkwh
        # self.infer_delta = self.predicted_co2eq_infer / self.actual_value - 1

        #print(self.predicted_co2eq_infer)


    def predict_num_throughput(self, node_size, memory_size):
        coeff_gpu = np.array([-2.12910565e-21, 4.39684339e-09, 7.99173057e02])
        coeff_batch = np.array([-4.29439186e-01, 5.21376002e01, 1.43737095e03])
        func_gpu = np.poly1d(coeff_gpu)
        func_batch = np.poly1d(coeff_batch)
        
        gpu_cap = 0.03 * memory_size
        
        parameter_num = 0
        
        if self.type == "MoE":
            parameter_num = self.base_param_b
        else:
            parameter_num = self.parameters_b
        
        if parameter_num < node_size * gpu_cap:
            p_size = 1
            t_size = int(np.ceil(parameter_num / gpu_cap))
        else:
            t_size = node_size
            p_size = int(np.ceil(parameter_num / (node_size * gpu_cap)))
        
        model_size = t_size * p_size
        
        num_gpu = (
            np.round(func_gpu(parameter_num * 1e9) / model_size) * model_size
        )
        
        if memory_size != 80:
            num_gpu = np.round(num_gpu * 2.5)
        if memory_size == 40:
            num_gpu *= 2
        
        d_size = num_gpu / model_size
        # estimated batch size
        if p_size == 1:
            batch_size = 512
        else:
            batch_size = np.round(func_batch(p_size) / 8) * 8
            
        if batch_size < num_gpu:
            batch_size = num_gpu
            
        coeff_tensor = np.array([-8.82079068e-20, 1.68591116e-09, 1.33954735e02])
        coeff_pipe = np.array([-5.60233749e-23, 8.45435587e-11, 1.34546129e02])
        func_tensor = np.poly1d(coeff_tensor)
        func_pipe = np.poly1d(coeff_pipe)
        rel_thru = 7.76 / 33.46
        
        # intra model condition
        if t_size <= node_size and p_size == 1:
            X = func_tensor(parameter_num * 1e9)
        # inter model
        else:
            X = func_pipe(parameter_num * 1e9)
            
        if memory_size != 80:
            X_new = X - X * rel_thru
            peak_new = X_new / 312
            self.achievedTFLOPs = peak_new * 125
        else:
            self.achievedTFLOPs = X
            
        self.device_num = num_gpu
