import os
from copy import deepcopy

import torch
from torch import nn
from dpipe.layers.resblock import ResBlock2d, ResBlock
from dpipe.layers.conv import PreActivation2d, PreActivationND


class UNet2D(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, kernel_size=3, padding=1, pooling_size=2, n_filters_init=8,
                 dropout=False, p=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_size = pooling_size
        n = n_filters_init
        if dropout:
            dropout_layer = nn.Dropout(p)
        else:
            dropout_layer = nn.Identity()

        self.policy_shape = 33
        self.policy_tracker = torch.zeros(self.policy_shape)
        self.policy_tracker_temp = torch.zeros(self.policy_shape)
        self.iter_tracker = 0
        self.iter_tracker_temp = 0

        self.parallelized_blocks = (nn.Conv2d, nn.ConvTranspose2d, ResBlock, PreActivationND)
        self.val_flag = False
        self.val_policy_tracker = torch.zeros(self.policy_shape)
        self.val_iter_tracker = 0

        self.init_path = nn.ModuleList([
            nn.Conv2d(n_chans_in, n, self.kernel_size, padding=self.padding, bias=False),
            nn.ReLU(),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding)
        ])
        self.shortcut0 = nn.Conv2d(n, n, 1)

        self.init_path_freezed = deepcopy(self.init_path)
        self.shortcut0_freezed = deepcopy(self.shortcut0)

        self.down1 = nn.ModuleList([
            nn.BatchNorm2d(n),
            nn.Conv2d(n, n * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding)
        ])
        self.shortcut1 = nn.Conv2d(n * 2, n * 2, 1)

        self.down1_freezed = deepcopy(self.down1)
        self.shortcut1_freezed = deepcopy(self.shortcut1)

        self.down2 = nn.ModuleList([
            nn.BatchNorm2d(n * 2),
            nn.Conv2d(n * 2, n * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding)
        ])
        self.shortcut2 = nn.Conv2d(n * 4, n * 4, 1)

        self.down2_freezed = deepcopy(self.down2)
        self.shortcut2_freezed = deepcopy(self.shortcut2)

        self.down3 = nn.ModuleList([
            nn.BatchNorm2d(n * 4),
            nn.Conv2d(n * 4, n * 8, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            dropout_layer
        ])

        self.down3_freezed = deepcopy(self.down3)

        self.up3 = nn.ModuleList([
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 8),
            nn.ConvTranspose2d(n * 8, n * 4, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        ])

        self.up3_freezed = deepcopy(self.up3)

        self.up2 = nn.ModuleList([
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 4),
            nn.ConvTranspose2d(n * 4, n * 2, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        ])

        self.up2_freezed = deepcopy(self.up2)

        self.up1 = nn.ModuleList([
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 2),
            nn.ConvTranspose2d(n * 2, n, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        ])

        self.up1_freezed = deepcopy(self.up1)

        self.out_path = nn.ModuleList([
            ResBlock2d(n, n, kernel_size=1),
            PreActivation2d(n, n_chans_out, kernel_size=1),
            nn.BatchNorm2d(n_chans_out)
        ])

        self.out_path_freezed = deepcopy(self.out_path)

    def forward_block(self, x, block_ft, block_fr, action_mask, i):

        for layer_ft, layer_fr in zip(block_ft, block_fr):
            policy_current = action_mask[..., i]
            x = layer_ft(x)*(1-policy_current) + layer_fr(x)*policy_current
            if isinstance(layer_ft, self.parallelized_blocks):
                i += 1
        return x, i

    def forward(self, x, policy):

        action = policy.contiguous()  # [32, 8]
        action_mask = action.view(-1, 1, 1, 1, action.shape[1])  # [32, 1, 1, 1, 8]
        i = 0

        x, i = self.forward_block(x, self.init_path, self.init_path_freezed, action_mask, i)
        shortcut0 = self.shortcut0(x)*(1-action_mask[..., self.policy_shape - 3]) + self.shortcut0_freezed(x) * \
                    action_mask[..., self.policy_shape - 3]

        x, i = self.forward_block(x, self.down1, self.down1_freezed, action_mask, i)
        shortcut1 = self.shortcut1(x) * (1 - action_mask[..., self.policy_shape - 2]) + self.shortcut1_freezed(x) * \
                    action_mask[..., self.policy_shape - 2]

        x, i = self.forward_block(x, self.down2, self.down2_freezed, action_mask, i)
        shortcut2 = self.shortcut2(x) * (1 - action_mask[..., self.policy_shape - 1]) + self.shortcut2_freezed(x) * \
                    action_mask[..., self.policy_shape - 1]

        x, i = self.forward_block(x, self.down3, self.down3_freezed, action_mask, i)
        x, i = self.forward_block(x, self.up3, self.up3_freezed, action_mask, i)
        x, i = self.forward_block(x + shortcut2, self.up2, self.up2_freezed, action_mask, i)
        x, i = self.forward_block(x + shortcut1, self.up1, self.up1_freezed, action_mask, i)
        x, i = self.forward_block(x + shortcut0, self.out_path, self.out_path_freezed, action_mask, i)

        if self.val_flag:
            self.val_iter_tracker += action.shape[0]  # batch size
            self.val_policy_tracker += torch.sum(action, dim=0).to('cpu')
        else:
            self.iter_tracker += action.shape[0]  # batch size
            self.policy_tracker += torch.sum(action, dim=0).to('cpu')

        return x

    def save_policy(self, folder_name):

        we_are_here = os.path.abspath('.')
        folder_to_store_in = os.path.join(we_are_here, folder_name)
        if not os.path.exists(folder_to_store_in):
            os.mkdir(folder_to_store_in)

        torch.save(self.policy_tracker, os.path.join(folder_to_store_in, 'policy_record'))

        f = open(os.path.join(folder_to_store_in, 'iter_record'), "w")
        f.write(str(self.iter_tracker))
        f.close()

        self.policy_tracker = torch.zeros(self.policy_shape)
        self.iter_tracker = 0

    def get_val_stats(self):

        val_stats = self.val_policy_tracker.detach().numpy() / self.val_iter_tracker

        self.val_policy_tracker = torch.zeros(self.policy_shape)
        self.val_iter_tracker = 0

        tb_record = {}
        for i in range(self.policy_shape-3):
            tb_record['val: block ' + str(i+1)] = val_stats[i]
        for i in range(self.policy_shape-3, self.policy_shape):
            tb_record['val: shortcut ' + str(i-(self.policy_shape-4))] = val_stats[i]

        return tb_record

    def get_train_stats(self):

        train_stats = (self.policy_tracker.detach().numpy() - self.policy_tracker_temp.detach().numpy()) / \
                      (self.iter_tracker - self.iter_tracker_temp)

        self.iter_tracker_temp = self.iter_tracker
        self.policy_tracker_temp = self.policy_tracker.clone()

        tb_record = {}
        for i in range(self.policy_shape-3):
            tb_record['train: block ' + str(i+1)] = train_stats[i]
        for i in range(self.policy_shape-3, self.policy_shape):
            tb_record['train: shortcut ' + str(i-(self.policy_shape-4))] = train_stats[i]

        return tb_record
