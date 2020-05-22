import nncf
from nncf import create_compressed_model, Config as NNCFConfig

# Instantiate your uncompressed model
from torchvision.models.resnet import resnet50
model = resnet50()

# Apply compression according to a loaded NNCF config
nncf_config = NNCFConfig.from_json("resnet50_int8.json")
comp_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual torch.nn.Module

# ... the rest of the usual PyTorch-powered training pipeline

# Export to ONNX or .pth when done fine-tuning
comp_ctrl.export_model("compressed_model.onnx")
torch.save(compressed_model.state_dict(), "compressed_model.pth")