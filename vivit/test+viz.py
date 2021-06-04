import torch
import sys
from vivit import ViViT
from utils import plotLearningCurve, decompress_pickle
import torch.nn.functional as F
from einops import reduce
from torchDataLoader import TorchDataLoader

CHECKPOINT_PATH = "checkpoints/"
PLOTS_PATH = "plots/"

def plot_loss_and_accuracy(train_epoch_loss, valid_loss, train_epoch_acc, valid_acc, name_prefix):
    obj = {'accuracy': train_epoch_acc, 'val_accuracy': valid_acc, 'loss': train_epoch_loss, 'val_loss': valid_loss}
    plotLearningCurve(obj, name_prefix=name_prefix+"_")

@torch.no_grad()
def eval_policy(model, valid_loader, device) -> float:
    model.eval()
    valid_loss = 0.0
    accuracy = 0
    
    count = 0
    for local_batch, local_labels in valid_loader:
        print(local_batch.shape)
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        local_batch = local_batch.permute(0, 1, 4, 3, 2)
        local_batch = F.pad(local_batch, (40,40))
        local_batch = reduce(local_batch, 'b f c (h 4) (w 4) -> b f c (h) (w)', 'mean')
        # print(local_batch.shape)
        out = model(local_batch)
        valid_loss += F.cross_entropy(out, local_labels)
        accuracy += (torch.softmax(out, dim=1).argmax(dim=1) == local_labels).sum().float() / float( local_labels.size(0) )
    print(count)
    print(accuracy)
    print(f"Validation loss: {valid_loss:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    return valid_loss, accuracy

def main(exp_name):
    device = torch.device("cuda")

    model = ViViT(80, 16, 101, 10, emb_dropout = 0.15).to(device)
    # get_total_params(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    checkpoint = torch.load(CHECKPOINT_PATH+exp_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_step_loss = checkpoint['train_step_loss']
    valid_loss = checkpoint['valid_loss']
    train_epoch_loss = checkpoint['train_epoch_loss']
    train_10_loss = checkpoint['train_10_loss']
    train_10_acc = checkpoint['train_10_acc']
    train_epoch_acc = checkpoint['train_epoch_acc']
    valid_acc = checkpoint['valid_acc']

    labels = decompress_pickle('../labels.pickle.pbz2')
    partition = decompress_pickle('../partition.pickle.pbz2')

    print(len(partition['val']))
    validation_dataset = TorchDataLoader(partition['val'], labels)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=1701, shuffle=True)

    eval_policy(model, validation_generator, device)

    # plot_loss_and_accuracy(train_epoch_loss, valid_loss, train_epoch_acc, valid_acc, PLOTS_PATH+exp_name)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Didn't find an experiment name. Exiting")
    else:
        main(sys.argv[1])