from typing import *

import torch
import torchvision
import lightning.pytorch as pl
from torch import nn
from torchmetrics.functional import jaccard_index
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights


class SegmentationModel(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_func: "Loss", config: Dict):  # noqa
        super().__init__()
        self.model = model
        self.loss_func = loss_func

        self.learning_rate = config["optimizer_params"]["lr"]
        self.betas = config["optimizer_params"]["betas"]
        self.weight_decay = config["optimizer_params"]["weight_decay"]

        self.n_classes = config["n_classes"]
        self.save_hyperparameters(ignore=["model", "loss_func"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        iou = jaccard_index(
            pred, target, task="multiclass",
            ignore_index=-1,  # -1 is non-labeled class
            num_classes=self.n_classes
        )
        self.log("train/IoU", iou, on_epoch=True, sync_dist=True)
        loss = self.loss_func(pred, target)
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        self.scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        iou = jaccard_index(
            pred, target, task="multiclass",
            ignore_index=-1,  # -1 is non-labeled class
            num_classes=self.n_classes
        )
        self.log("validation/IoU", iou, on_epoch=True, sync_dist=True)
        loss = self.loss_func(pred, target)
        self.log("validation/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        iou = jaccard_index(
            pred, target, task="multiclass",
            ignore_index=-1,  # -1 is non-labeled class
            num_classes=self.n_classes
        )
        self.log("validation/IoU", iou, on_epoch=True, sync_dist=True)
        return {'IoU': iou}

    def configure_optimizers(self):
        # The Optimizer and Scheduler is set in accordance with the article.
        # see https://www.mosaicml.com/blog/behind-the-scenes
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.betas,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            # Maximum number of iterations.
            T_max=self.trainer.estimated_stepping_batches,
            # Minimum learning rate is 0, at that point we will stop the training.
            eta_min=0,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            # "interval": "step"  it doesn't work... so I update scheduler in training_step
        }


class ConvNet(nn.Module):
    def __init__(
        self,
        num_class: int = 3,
        dropout: float = 0.1,
        use_deform_convs: Optional[str] = None,  # v1 or v2
    ):
        super().__init__()
        self.modified_resnet = self._build_resnet(
            num_class=num_class,
            dropout=dropout,
            use_deform_convs=use_deform_convs,
        )

    def _build_resnet(self, num_class: int, dropout: float, use_deform_convs: Optional[str] = None) -> nn.Module:
        resnet = fcn_resnet50(backbone_weights=ResNet50_Weights.IMAGENET1K_V2, num_classes=num_class)
        resnet.classifier[3].p = dropout

        # One option is to freeze the first few layers and do transfer learning,
        # but on my experiments is does not impact much.
        # Freeze the first two layers.
        # for param in resnet.backbone.layer1.parameters():
        #     param.requires_grad = False
        # for param in resnet.backbone.layer2.parameters():
        #     param.requires_grad = False
        # for param in resnet.backbone.layer3[0:3].parameters():
        #     param.requires_grad = False

        if use_deform_convs is None:
            return resnet

        if use_deform_convs == 'v1':
            modulation = False
        elif use_deform_convs == 'v2':
            modulation = True
        else:
            raise ValueError('Argument `use_deform_convs` should be: None, "v1" or "v2"')

        # Replacing 3x3 Conv blocks as it was stated in paper.
        # Default combination is to replace last three Conv2d 3x3 layers to DeformableConv2d.
        for layers, indexes in [
            # (resnet.backbone.layer3, (3, 4, 5)),
            (resnet.backbone.layer4, (0, 1, 2)),
        ]:
            for index in indexes:
                conv = layers[index].conv2
                deform_conv_layer = DeformableConv2d(
                    conv.in_channels,
                    conv.out_channels,
                    kernel_size=conv.kernel_size[0],
                    stride=conv.stride[0],
                    dilation=conv.dilation,
                    bias=conv.bias,
                    modulation=modulation
                )
                # We load pretrained weights as initialization for deformable convolutions.
                # Offsets are sets to zero.
                deform_conv_layer.deform_conv.load_state_dict(conv.state_dict())
                layers[index].conv2 = deform_conv_layer
        return resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modified_resnet(x)["out"]


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        modulation: bool = False,
    ):
        super(DeformableConv2d, self).__init__()
        # offsets to be applied for each position in the convolution kernel.
        self.offset_channel = 2 * kernel_size ** 2
        # masks to be applied for each position in the convolution kernel.
        self.mask_channel = kernel_size ** 2

        self.modulation = modulation
        if self.modulation:
            offset_out_channels = self.offset_channel + self.mask_channel
        else:
            offset_out_channels = self.offset_channel

        self.offset_conv = torch.nn.Conv2d(
            in_channels,
            offset_out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True
        )
        # In the training, these added conv and fc layers for offset learning are initialized with zero weights
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modulation:
            offset_and_mask = self.offset_conv(x)
            offset, mask = torch.split(
                offset_and_mask,
                split_size_or_sections=[self.offset_channel, self.mask_channel],
                dim=1,
            )
            mask = torch.sigmoid(mask)
        else:
            offset = self.offset_conv(x)
            mask = None
        x = self.deform_conv(x, offset, mask=mask)
        return x
