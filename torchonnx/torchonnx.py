import numpy as np
import onnx
import os
import glob
import onnx_backend as backend

from onnx import numpy_helper

model = onnx.load("model.onnx")
test_data_dir = "test_data_set_0"

# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, "input_{}.pb".format(i))
    tensor = onnx.TensorProto()
    with open(input_file, "rb") as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, "output_*.pb")))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, "output_{}.pb".format(i))
    tensor = onnx.TensorProto()
    with open(output_file, "rb") as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# Run the model on the backend
outputs = list(backend.run_model(model, inputs))

# Compare the results with reference outputs.
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o)