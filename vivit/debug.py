import torch
from torchDataLoader import TorchDataLoader
from utils import compressed_pickle, decompress_pickle, plotLearningCurve
from vivit import ViViT
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU {torch.cuda.get_device_name(device)}.")
else:
    print("No GPU found. Falling back to CPU.")
    device = torch.device("cpu")

model = ViViT(224, 16, 100, 16).to(device)
print(model)
# get_total_params(model)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-5,
)

img = torch.ones([1, 16, 3, 224, 224])
label = torch.ones([1], dtype=torch.long)

complete = False
global_step = 0
epoch = 0
try:
    
    
    local_batch, local_labels = img.to(device), label.to(device)

    model.train()
    optimizer.zero_grad()
    out = model(local_batch)
    loss = F.nll_loss(out, local_labels)
    loss.backward()
    
    optimizer.step()

except KeyboardInterrupt:
    print("Caught keyboard interrupt. Saving model before quitting.")