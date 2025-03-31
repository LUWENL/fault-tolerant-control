import time
import torch


def number_of_parameters(model):
    # unit: K
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1000


def size_of_model(model):
    # unit: Byte
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())

    return model_size / 1024


def inference_time(model):
    # unit: ms


    start_time = time.time()

    inputs = torch.randn(1, 50, 6).to('cuda')
    output = model(inputs)

    end_time = time.time()
    consumed_time = end_time - start_time

    print(consumed_time , "seconds")
    return consumed_time * 1000


def all_metrics(model):
    print("number of parameters: {} K".format(number_of_parameters(model)))
    print("size of model: {} KB".format(size_of_model(model)))
    print("inference time: {} ms".format(inference_time(model)))
