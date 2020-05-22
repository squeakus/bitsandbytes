import torchvision
import torch
from torch import nn

model_conv=torchvision.models.alexnet(pretrained=True)

# change the first layer
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, n_class)

# for name, child in model_conv.named_children():
#     for name2, params in child.named_parameters():
#         print(name, name2)

# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in model_conv.state_dict():
#     print(var_name, "\t", model_conv.state_dict()[var_name])

# # Freeze the weights
# ct = 0
# for name, child in model_conv.named_children():
#     ct += 1
#     if ct < 7:
#         for name2, params in child.named_parameters():
#         	params.requires_grad = False

torch.save(model_conv,"resnet50.pth")

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
dummy_input = torch.randn(10, 3, 224, 224)

torch.onnx.export(model_conv, dummy_input, "resnet50.onnx", 
				  verbose=False, input_names=input_names,
				  output_names=output_names)
# Model class must be defined somewhere
model = torch.load("resnet50.pth")
model.eval()