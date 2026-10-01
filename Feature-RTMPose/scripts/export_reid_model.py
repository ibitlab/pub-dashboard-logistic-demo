"""One-time export: OSNet-AIN PyTorch -> ONNX for person re-identification.

Run this once to generate the ONNX model, then deploy with onnxruntime only.
The exported model is placed in data/models/osnet_ain_x1_0.onnx (+ .onnx.data weights)

Requirements (temporary, only for export):
    pip install torch torchvision torchreid onnx onnxscript onnxruntime Pillow tensorboard \
                scipy opencv-python-headless matplotlib h5py yacs
    (torchreid imports the second line too). Use a throwaway venv.

After export, these packages can be removed from the production environment.
"""

import os
import sys

def main():
    try:
        import torch
        import torchreid
        import onnxruntime as ort
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch torchvision torchreid onnx onnxscript onnxruntime Pillow tensorboard scipy opencv-python-headless matplotlib h5py yacs")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "osnet_ain_x1_0.onnx")

    if os.path.exists(output_path):
        print(f"Model already exists: {output_path}")
        print("Delete it first if you want to re-export.")
        return

    print("Building OSNet-AIN x1.0 (pretrained on ImageNet)...")
    model = torchreid.models.build_model(
        name="osnet_ain_x1_0",
        num_classes=1,
        loss="softmax",
        pretrained=True,
    )
    model.eval()

    print("Exporting to ONNX...")
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=11,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    )

    # Verify
    session = ort.InferenceSession(output_path)
    out = session.run(None, {"input": np.random.randn(1, 3, 256, 128).astype(np.float32)})
    size_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"Output shape: {out[0].shape} (expected: (1, 512))")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Saved to: {output_path}")
    print("DONE. You can now remove torch/torchreid from your environment.")


if __name__ == "__main__":
    main()
