import torch
from torchDataLoader import TorchDataLoader
from utils import compressed_pickle, decompress_pickle, plotLearningCurve
from vivit import ViViT
from einops import reduce
import torch.nn.functional as F
import sys
#from torchkit import Logger, checkpoint
import time

CHECKPOINT_PATH = "checkpoints/"
@torch.no_grad()
def eval_policy(model, valid_loader, device) -> float:
    model.eval()
    valid_loss = 0.0
    accuracy = 0
    
    for local_batch, local_labels in valid_loader:
        # print(local_batch.shape)
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        local_batch = local_batch.permute(0, 1, 4, 3, 2)
        local_batch = F.pad(local_batch, (40,40))
        local_batch = reduce(local_batch, 'b f c (h 4) (w 4) -> b f c (h) (w)', 'mean')
        # print(local_batch.shape)
        out = model(local_batch)
        valid_loss = F.cross_entropy(out, local_labels)
        accuracy = (torch.softmax(out, dim=1).argmax(dim=1) == local_labels).sum().float() / float( local_labels.size(0) )
        break
    print(f"Validation loss: {valid_loss:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    return valid_loss, accuracy

def main(exp_name, resume, remove_positional, remove_layernorm, remove_space, remove_temporal, remove_dropout):
    #logger = Logger("./logger")
    # Setup compute device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU {torch.cuda.get_device_name(device)}.")
    else:
        print("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    labels = decompress_pickle('../labels.pickle.pbz2')
    partition = decompress_pickle('../partition.pickle.pbz2')

    # print(labels)
    # print(partition)
    training_dataset = TorchDataLoader(partition['train'], labels)
    validation_dataset = TorchDataLoader(partition['val'], labels)
    training_generator = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

    num_train_pairs = len(training_generator.dataset)
    num_valid_pairs = len(validation_generator.dataset)
    print(f"Training on {num_train_pairs} state, action pairs.")
    print(f"Validating on {num_valid_pairs} state, action pairs.")

    model = ViViT(80, 16, 101, 10, emb_dropout = 0.15,
                remove_positional=remove_positional,
                remove_layernorm=remove_layernorm,
                remove_space=remove_space,
                remove_temporal=remove_temporal,
                remove_dropout=remove_dropout).to(device)
    # get_total_params(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )

    # Create checkpoint manager.
    # checkpoint_dir = "../checkpoints/vivit"
    # checkpoint_manager = checkpoint.CheckpointManager(
    #     checkpoint.Checkpoint(policy=model, optimizer=optimizer),
    #     checkpoint_dir,
    #     device,
    # )

    complete = False
    global_step = 0
    epoch = 0

    train_step_loss = []
    valid_loss = []
    train_epoch_loss = []
    train_10_loss = []
    train_10_acc = []
    train_epoch_acc = []
    valid_acc = []

    if resume:
        checkpoint = torch.load(CHECKPOINT_PATH+exp_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        train_step_loss = checkpoint['train_step_loss']
        valid_loss = checkpoint['valid_loss']
        train_epoch_loss = checkpoint['train_epoch_loss']
        train_10_loss = checkpoint['train_10_loss']
        train_10_acc = checkpoint['train_10_acc']
        train_epoch_acc = checkpoint['train_epoch_acc']
        valid_acc = checkpoint['valid_acc']

    try:
        while not complete:
            t0 = time.time()
            t1 = time.time()
            for local_batch, local_labels in training_generator:
                # print(local_batch.shape)
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                local_batch = local_batch.permute(0, 1, 4, 3, 2).to(device)
                local_batch = F.pad(local_batch, (40,40)).to(device)
                local_batch = reduce(local_batch, 'b f c (h 4) (w 4) -> b f c (h) (w)', 'mean').to(device)
                # print(local_batch.shape)
                model.train()
                optimizer.zero_grad()
                out = model(local_batch)
                loss = F.cross_entropy(out, local_labels)
                loss.backward()

                train_step_loss.append(loss)
                
                optimizer.step()

                if not global_step % 10:
                    #logger.log_scalar(loss, global_step, "train/loss")
                    train_10_loss.append(loss)
                    accuracy = (torch.softmax(out, dim=1).argmax(dim=1) == local_labels).sum().float() / float( local_labels.size(0) )
                    train_10_acc.append(accuracy)

                    if not global_step % 50:
                        train_epoch_acc.append(accuracy)
                    #logger.log_scalar(accuracy, global_step, "train/accuracy")
                    print(
                        "Iter[{}/{}] (Epoch {}), Loss: {:.3f}, Accuracy: {:.6f}".format(
                            global_step,
                            10000,
                            epoch,
                            loss.item(),
                            accuracy
                        )
                    )
                    t1 = time.time()

                    total = t1-t0

                    
                    print(f"Took {total} seconds to run the last 10 iterations")

                if not global_step % 50:
                    valid_curr_loss, valid_curr_accuracy = eval_policy(model, validation_generator, device)
                    valid_loss.append(valid_curr_loss)
                    train_epoch_loss.append(loss)
                    valid_acc.append(valid_curr_accuracy)
                    
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_step_loss': train_step_loss,
                        'valid_loss': valid_loss,
                        'train_epoch_loss': train_epoch_loss,
                        'train_10_acc': train_10_acc,
                        'train_epoch_acc': train_epoch_acc,
                        'valid_acc': valid_acc,
                        'train_10_loss': train_10_loss,
                    }, CHECKPOINT_PATH+exp_name)

                    #logger.log_scalar(valid_loss, global_step, "valid/loss")
                    #logger.log_scalar(valid_accuracy, global_step, "valid/accuracy")

                if not global_step % 10:
                    t0 = time.time()
                    

                # Save model checkpoint.
                # if not global_step % 10:
                #     checkpoint_manager.save(global_step)

                # Exit if complete.
                global_step += 1
                if global_step > 800:
                    complete = True
                    break
            epoch += 1

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Saving model before quitting.")

    finally:
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_step_loss': train_step_loss,
            'valid_loss': valid_loss,
            'train_epoch_loss': train_epoch_loss,
            'train_10_acc': train_10_acc,
            'train_epoch_acc': train_epoch_acc,
            'valid_acc': valid_acc,
            'train_10_loss': train_10_loss,
        }, CHECKPOINT_PATH+exp_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Didn't find an experiment name. Exiting")
    else:
        resume = False
        remove_positional=False
        remove_layernorm=False
        remove_space=False
        remove_temporal=False
        remove_dropout=False

        if "resume" in sys.argv:
            resume = True
        
        if "remove_positional" in sys.argv:
            remove_positional = True

        if "remove_layernorm" in sys.argv:
            remove_layernorm = True
        
        if "remove_space" in sys.argv:
            remove_space = True

        if "remove_temporal" in sys.argv:
            remove_temporal = True

        if "remove_dropout" in sys.argv:
            remove_dropout = True
        
        main(sys.argv[1], resume, remove_positional, remove_layernorm, remove_space, remove_temporal, remove_dropout)