from model.transformer import (TransformerPretrain)
import os
import csv
import yaml
import shutil
import argparse
import torch.nn as nn
from model.utils import *
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import CIFData
from dataset.dataset import collate_pool, get_train_val_test_loader
from model.cgcnn import CrystalGraphConvNet
import sys


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class Model(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len):
        super(Model, self).__init__()

        self.model_t = TransformerPretrain(d_model= 512, nhead= 4, d_hid= 512, nlayers= 3, dropout= 0.1)
        self.model_g = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len= 64, h_fea_len=128, n_conv=3, n_h=1)

        self.g = nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320, 1),
        )


    def forward(self, src, frac,input_graph):
        zjs = self.model_g(*input_graph)
        zis = self.model_t(src, frac)
        z = torch.cat((zjs, zis), dim=1)
        return self.g(z)


class FineTune(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = nn.L1Loss()
        self.fudge = 0.02
        self.traindata = CIFData(self.config['task'],**self.config['traindata'])
        self.valid = CIFData(self.config['task'],**self.config['validdata'])
        self.testdata = CIFData(self.config['task'], **self.config['testdata'])

        self.dataset = CIFData(self.config['task'],**self.config['dataset'])
        self.random_seed = self.config['random_seed']
        collate_fn = collate_pool
        self.train_loader = get_train_val_test_loader(
            dataset=self.traindata,
            collate_fn=collate_fn,
            pin_memory=self.config['gpu'] != 'cpu',
            batch_size=self.config['batch_size'],
            return_test=True,
            **self.config['dataloader']
        )
        self.valid_loader = get_train_val_test_loader(
            dataset=self.valid,
            collate_fn=collate_fn,
            pin_memory=self.config['gpu'] != 'cpu',
            batch_size=self.config['batch_size'],
            return_test=True,
            **self.config['dataloader']
        )
        self.test_loader = get_train_val_test_loader(
            dataset=self.testdata,
            collate_fn=collate_fn,
            pin_memory=self.config['gpu'] != 'cpu',
            batch_size=self.config['batch_size'],
            return_test=True,
            **self.config['dataloader']
        )

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def train(self):
        structures, _, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = Model(orig_atom_fea_len, nbr_fea_len).to(self.device)
        model = self._load_pre_trained_weights(model)

        if self.config['cuda']:
            model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['optim']['lr'], weight_decay=eval(self.config['optim']['weight_decay']))
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')


        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_mae = np.inf

        epochs_no_improve = 0  
        patience = 20

        for epoch_counter in range(self.config['epochs']):
            total_loss = 0.0
            for bn, (input_1,input_2, target, _) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                 Variable(input_1[1].to(self.device, non_blocking=True)),
                                 input_1[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])
                    input_transformer = input_2.to(self.device, non_blocking=True)
                    src, frac = input_transformer.squeeze(-1).chunk(2, dim=1)
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
                    frac = torch.clamp(frac, 0, 1)
                    frac[src == 0] = 0
                    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

                    src = src.to(self.device, dtype=torch.long, non_blocking=True)
                    frac = frac.to(self.device, dtype=torch.float32, non_blocking=True)
                else:
                    input_graph = (Variable(input_1[0]),
                                 Variable(input_1[1]),
                                 input_1[2],
                                 input_1[3])
                    input_transformer = input_2
                    src, frac = input_transformer.squeeze(-1).chunk(2, dim=1)
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
                    frac = torch.clamp(frac, 0, 1)
                    frac[src == 0] = 0
                    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

                    src = src.to(dtype=torch.long, non_blocking=True)
                    frac = frac.to(dtype=torch.float32, non_blocking=True)

                target_normed = target

                if self.config['cuda']:
                   target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                   target_var = Variable(target_normed)

                output = model(src, frac, input_graph)


                loss = self.criterion(output, target_var)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch_counter + 1}, Average Loss: {avg_loss:.4f}')       

            self.writer.add_scalar('train_loss', avg_loss, global_step=n_iter)
      

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                if valid_mae < best_valid_mae:
                    # save the model weights
                    best_valid_mae = valid_mae
                    epochs_no_improve = 0  
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                else:  
                    epochs_no_improve += 1  

                   

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

    
            if epochs_no_improve == patience:  
                  print(f'Early stopping')  
                  break 
  



        self.model = model
        
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = self.config['fine_tune_from']
            print(os.path.join(checkpoints_folder, 'model.pth'))
            load_state = torch.load(os.path.join(checkpoints_folder, 'model_146.pth'), map_location=self.config['gpu'])

            model_state = model.state_dict()

            for name, param in load_state.items():
                if name not in model_state:
                    print('NOT loaded:', name)
                    continue
                else:
                    print('loaded:', name)
                if isinstance(param, nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[name].copy_(param)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()


        with torch.no_grad():
            model.eval()

            for bn, (input_1,input_2, target, _) in enumerate(valid_loader):
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                 Variable(input_1[1].to(self.device, non_blocking=True)),
                                 input_1[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])
                    input_transformer = input_2.to(self.device, non_blocking=True)
                    src, frac = input_transformer.squeeze(-1).chunk(2, dim=1)
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
                    frac = torch.clamp(frac, 0, 1)
                    frac[src == 0] = 0
                    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

                    src = src.to(self.device, dtype=torch.long, non_blocking=True)
                    frac = frac.to(self.device, dtype=torch.float32, non_blocking=True)
                else:
                    input_graph = (Variable(input_1[0]),
                                 Variable(input_1[1]),
                                 input_1[2],
                                 input_1[3])
                    input_transformer = input_2
                    src, frac = input_transformer.squeeze(-1).chunk(2, dim=1)
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
                    frac = torch.clamp(frac, 0, 1)
                    frac[src == 0] = 0
                    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

                    src = src.to(dtype=torch.long, non_blocking=True)
                    frac = frac.to(dtype=torch.float32, non_blocking=True)

                target_normed = target
                # target_normed = self.normalizer.norm(target)

                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                output = model(src, frac, input_graph)

                loss = self.criterion(output, target_var)

                
                mae_error = mae(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))




            print('Epoch [{0}] Validate: [{1}/{2}], '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                n_epoch + 1, bn + 1, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))

           


        model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

    def test(self):
        # test steps
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        mae_errors = AverageMeter()

        test_targets = []
        test_preds = []
        test_cif_ids = []

        with torch.no_grad():
            self.model.eval()
            for bn, (input_1,input_2, target, batch_cif_ids) in enumerate(self.test_loader):
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                 Variable(input_1[1].to(self.device, non_blocking=True)),
                                 input_1[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])
                    input_transformer = input_2.to(self.device, non_blocking=True)
                    src, frac = input_transformer.squeeze(-1).chunk(2, dim=1)
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
                    frac = torch.clamp(frac, 0, 1)
                    frac[src == 0] = 0
                    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

                    src = src.to(self.device, dtype=torch.long, non_blocking=True)
                    frac = frac.to(self.device, dtype=torch.float32, non_blocking=True)

                else:
                    input_graph = (Variable(input_1[0]),
                                 Variable(input_1[1]),
                                 input_1[2],
                                 input_1[3])
                    input_transformer = input_2
                    src, frac = input_transformer.squeeze(-1).chunk(2, dim=1)
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
                    frac = torch.clamp(frac, 0, 1)
                    frac[src == 0] = 0
                    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

                    src = src.to(dtype=torch.long, non_blocking=True)
                    frac = frac.to(dtype=torch.float32, non_blocking=True)

                target_normed = target

                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                output = self.model(src, frac, input_graph)

                loss = self.criterion(output, target_var)

                mae_error = mae(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                test_pred = output.data.cpu()
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

            print('Test: [{0}/{1}], '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))

        with open(os.path.join(self.writer.log_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target_var, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target_var, pred))



        self.model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
    parser.add_argument('--seed', default=1, type=int,
                        metavar='Seed', help='random seed for splitting data (default: 1)')

    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    config['random_seed'] = args.seed
    name= config['times']
    seed = config['random_seed']

    log_dir = os.path.join(
        'training_results/train',
        '{}'.format(seed)
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fine_tune = FineTune(config, log_dir)
    fine_tune.train()
    loss, metric = fine_tune.test()






