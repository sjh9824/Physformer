import torch

print(torch.cuda.is_available())

device = torch.device('cuda:0')

print(torch.cuda.get_device_name(device = 0))
print(torch.cuda.device_count())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)