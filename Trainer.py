import os
import pytz
from datetime import datetime
import random
import argparse
import numpy as np
from tqdm import tqdm
import imageio
from math import log10
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import stack
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from dataloader import Dataset_Dance
from modules import (
    Generator,
    Gaussian_Predictor,
    Decoder_Fusion,
    Label_Encoder,
    RGB_Encoder,
)


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio

        self.beta = 0.001
        if self.kl_anneal_type == "Without":
            self.beta = 1.0

    def update(self, current_epoch):
        if self.kl_anneal_type == "Cyclical":
            self.beta = self.frange_cycle_linear(
                current_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=self.kl_anneal_cycle,
                ratio=self.kl_anneal_ratio,
            )
        elif self.kl_anneal_type == "Monotonic":
            self.beta = self.frange_monotonic(
                current_epoch,
                start=0.0,
                stop=1.0,
                n_cycle=self.kl_anneal_cycle,
                ratio=self.kl_anneal_ratio,
            )

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        new_beta = (n_iter % n_cycle) / n_cycle * ratio
        new_beta = round(new_beta, 5)
        new_beta = min(new_beta, stop)

        return new_beta

    def frange_monotonic(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        delta = ratio / n_cycle

        new_beta = round(n_iter * delta, 5)
        new_beta = min(new_beta, stop)

        return new_beta


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # tensorboard writer
        if not self.args.test:
            self.writer = SummaryWriter(log_dir=f"{args.save_root}/logs")

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[2, 5], gamma=0.1
        )
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, x_frame, y_pose, y_frame):
        # train posterior predictor (y_frame: t frame, y_pose: t pose)
        y_frame_feature = self.frame_transformation(y_frame)
        y_pose_feature = self.label_transformation(y_pose)
        z, mu, logvar = self.Gaussian_Predictor(y_frame_feature, y_pose_feature)

        # train generator (x_frame: t-1 frame, y_pose: t pose, z: noise vector)
        x_frame_feature = self.frame_transformation(x_frame)
        decoder_output = self.Decoder_Fusion(x_frame_feature, y_pose_feature, z)

        # predicted y frame (predicted t frame)
        pred_frame = self.Generator(decoder_output)

        return mu, logvar, pred_frame

    def forward_inference(self, x_frame, y_pose):
        # noise vector
        z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W)
        z = z.to(self.args.device)

        # val generator (x_frame: t-1 frame, y_pose: t pose, z: noise vector)
        x_frame_feature = self.frame_transformation(x_frame)
        y_pose_feature = self.label_transformation(y_pose)

        decoder_output = self.Decoder_Fusion(x_frame_feature, y_pose_feature, z)

        # predicted y frame (predicted t frame)
        pred_frame = self.Generator(decoder_output)

        return pred_frame

    def training_stage(self):
        # dataloader
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()

        # best val psnr
        best_val_psnr = None

        for i in range(self.args.num_epoch):
            # update trainloader to full data (end fast train)
            if self.args.fast_train and self.current_epoch > self.args.fast_train_epoch:
                self.args.fast_train = False
                train_loader = self.train_dataloader()

            print(f"Epoch: {self.current_epoch}, train loader len: {len(train_loader)}")
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            # loss
            sum_loss = 0.0
            sum_kl_loss = 0.0
            sum_mse_loss = 0.0

            for img, label in (pbar := tqdm(train_loader)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)

                loss, kl_loss, mse_loss = self.training_one_step(
                    img, label, adapt_TeacherForcing
                )
                sum_loss += loss.item() * img.size(1)
                sum_kl_loss += kl_loss.item() * img.size(1)
                sum_mse_loss += mse_loss.item() * img.size(1)

                if adapt_TeacherForcing:
                    self.tqdm_bar(
                        "train [TeacherForcing: ON, {:.1f}], beta: {}".format(
                            self.tfr, self.kl_annealing.get_beta()
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )
                else:
                    self.tqdm_bar(
                        "train [TeacherForcing: OFF, {:.1f}], beta: {}".format(
                            self.tfr, self.kl_annealing.get_beta()
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )

            epoch_loss = sum_loss / len(train_loader.dataset)
            epoch_kl_loss = sum_kl_loss / len(train_loader.dataset)
            epoch_mse_loss = sum_mse_loss / len(train_loader.dataset)

            # validation
            epoch_val_loss, epoch_val_psnr, _ = self.evaluate(val_loader)

            # best psnr
            if best_val_psnr is None or epoch_val_psnr > best_val_psnr:
                best_val_psnr = epoch_val_psnr
                self.save(os.path.join(self.args.save_root, "ckpt", "epoch=best.ckpt"))

            # tensorboard logger
            self.writer.add_scalar("Loss/train", epoch_loss, self.current_epoch)
            self.writer.add_scalar("Loss/train/KL", epoch_kl_loss, self.current_epoch)
            self.writer.add_scalar("Loss/train/MSE", epoch_mse_loss, self.current_epoch)
            self.writer.add_scalar("Loss/val", epoch_val_loss, self.current_epoch)
            self.writer.add_scalar("PSNR/val", epoch_val_psnr, self.current_epoch)
            self.writer.add_scalar(
                "KL Annealing/beta", self.kl_annealing.get_beta(), self.current_epoch
            )
            self.writer.add_scalar(
                "TeacherForcing/tfr",
                self.tfr if adapt_TeacherForcing else 0.0,
                self.current_epoch,
            )

            # save ckpt
            if self.current_epoch % self.args.per_save == 0:
                self.save(
                    os.path.join(
                        self.args.save_root, "ckpt", f"epoch={self.current_epoch}.ckpt"
                    )
                )

            # update parameters
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update(self.current_epoch)

        print(f"Best PSNR: {best_val_psnr}")

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # enable training mode
        self.frame_transformation.train()
        self.label_transformation.train()
        self.Gaussian_Predictor.train()
        self.Decoder_Fusion.train()
        self.Generator.train()

        # loss
        kl_loss = 0.0
        mse_loss = 0.0

        # store the t-1 predicted frame for t prediction
        previous_frame = img[:, 0]

        # training
        for i in range(img.size(1) - 1):
            x_frame, y_frame = previous_frame, img[:, i + 1]
            y_pose = label[:, i + 1]

            # adapt_TeacherForcing: use ground truth frame t for t+1 prediction
            if adapt_TeacherForcing:
                x_frame = img[:, i]

            # forward
            mu, logvar, pred_frame = self.forward(x_frame, y_pose, y_frame)

            # loss
            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            mse_loss += self.mse_criterion(pred_frame, y_frame)

            # update previous frame for t+1 prediction
            previous_frame = pred_frame

        # kl annealing
        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kl_loss

        # backward
        loss.backward()

        # optimizer
        self.optimizer_step()
        self.optim.zero_grad()

        return loss, kl_loss, mse_loss

    @torch.no_grad()
    def evaluate(self, val_loader):
        for img, label in val_loader:
            img = img.to(self.args.device)
            label = label.to(self.args.device)

            loss, psnr, per_frame_psnr = self.val_one_step(img, label)

        return loss, psnr, per_frame_psnr

    def val_one_step(self, img, label):
        # enable eval mode
        self.frame_transformation.eval()
        self.label_transformation.eval()
        self.Gaussian_Predictor.eval()
        self.Decoder_Fusion.eval()
        self.Generator.eval()

        # mse loss
        mse_loss = 0.0

        # psnr
        sum_psnr = 0.0
        per_frame_psnr = []

        predicted_frames = []
        predicted_frames.append(img[:, 0])

        for i in (pbar := tqdm(range(img.size(1) - 1), ncols=100)):
            x_frame, y_frame = predicted_frames[-1], img[:, i + 1]
            y_pose = label[:, i + 1]

            pred_frame = self.forward_inference(x_frame, y_pose)
            mse_loss += self.mse_criterion(pred_frame, y_frame)

            predicted_frames.append(pred_frame)

            # psnr
            frame_psnr = Generate_PSNR(pred_frame, y_frame).item()
            per_frame_psnr.append(frame_psnr)
            sum_psnr += frame_psnr

            # last iter
            if i == img.size(1) - 2:
                self.tqdm_bar(
                    "val",
                    pbar,
                    mse_loss.detach().cpu(),
                    lr=self.scheduler.get_last_lr()[0],
                )

        avg_psnr = sum_psnr / (img.size(1) - 1)

        return mse_loss, avg_psnr, per_frame_psnr

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list,
            save_all=True,
            duration=40,
            loop=0,
        )

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )

        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="train",
            video_len=self.train_vi_len,
            partial=args.fast_partial if self.args.fast_train else args.partial,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="val",
            video_len=self.val_vi_len,
            partial=1.0,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return val_loader

    def teacher_forcing_ratio_update(self):
        # start lowering tfr after tfr_sde epoch
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(self.tfr - self.tfr_d_step, 0.0)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False
        )
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
            },
            path,
        )
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            self.args.lr = checkpoint["lr"]
            self.tfr = checkpoint["tfr"]

            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=[2, 4], gamma=0.1
            )
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint["last_epoch"]
            )
            self.current_epoch = checkpoint["last_epoch"]

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()


def plot_per_frame_psnr(per_frame_psnr, save_root):
    avg_psnr = round(np.mean(per_frame_psnr), 2)

    plt.figure(figsize=(8, 5))
    plt.plot(per_frame_psnr, label=f"PSNR: {avg_psnr}")
    plt.xlabel("Frame")
    plt.ylabel("PSNR")
    plt.title("Per Frame PSNR")
    plt.legend()

    plt.savefig(f"{save_root}/per_frame_psnr.png")


def create_output_dir(save_root):
    # create output(save_root) directory
    if save_root is not None:
        tz = pytz.timezone("Asia/Taipei")
        now = datetime.now(tz).strftime("%Y%m%d-%H%M")

        save_root = os.path.join(save_root, now)

        # create output directory
        os.makedirs(save_root, exist_ok=True)

        # create ckpt directory
        os.makedirs(f"{save_root}/ckpt", exist_ok=True)

        # create logs directory
        os.makedirs(f"{save_root}/logs", exist_ok=True)

        return save_root

    return None


def main(args):
    # create output dir in training mode
    if not args.test and args.save_root is not None:
        args.save_root = create_output_dir(args.save_root)

        # save all args in text
        with open(f"{args.save_root}/args.txt", "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()

    if args.test:
        val_loader = model.val_dataloader()
        loss, psnr, per_frame_psnr = model.evaluate(val_loader)
        plot_per_frame_psnr(per_frame_psnr, args.save_root)
        print(f"Validation Loss: {loss}, PSNR: {psnr}")
    else:
        model.training_stage()


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=16)
    parser.add_argument('--num_epoch',     type=int, default=30,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, choices=["Cyclical", "Monotonic", "Without"],  default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")

    # fmt: on

    args = parser.parse_args()

    main(args)
