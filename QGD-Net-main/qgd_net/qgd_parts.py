import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvalidKernelShape(RuntimeError):
    """Base class to generate custom exception if generating kernel failed."""

    def __init__(self, error_message):
        """Construct custom error with custom error message.
        :param error_message: The custom error message.
        """
        super().__init__(error_message)


class InvalidInput(RuntimeError):
    """Base class to generate custom exception if input is invalid."""

    def __init__(self, error_message):
        """Construct custom error with custom error message.
        :param error_message: The custom error message.
        """
        super().__init__(error_message)


class QuaternionLin(nn.Module):
    """Reproduction class of the quaternion linear layer."""

    def __init__(self, in_channels, out_channels, dimension=2, bias=True):
        """Create the quaterion linear layer."""
        super(QuaternionLin, self).__init__()

        self.in_channels = np.floor_divide(in_channels, 4)
        self.out_channels = np.floor_divide(out_channels, 4)

        self.weight_shape = self.get_weight_shape(self.in_channels, self.out_channels)
        self._weights = self.weight_tensors(self.weight_shape)

        self.r_weight, self.k_weight, self.i_weight, self.j_weight = self._weights

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.constant_(self.bias, 0)

    def forward(self, input_x):
        """Apply forward pass of input through quaternion linear layer."""
        cat_kernels_4_r = torch.cat(
            [self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=0
        )
        cat_kernels_4_i = torch.cat(
            [self.i_weight, self.r_weight, -self.k_weight, self.j_weight], dim=0
        )
        cat_kernels_4_j = torch.cat(
            [self.j_weight, self.k_weight, self.r_weight, -self.i_weight], dim=0
        )
        cat_kernels_4_k = torch.cat(
            [self.k_weight, -self.j_weight, self.i_weight, self.r_weight], dim=0
        )

        cat_kernels_4_quaternion = torch.cat(
            [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1
        )

        if self.bias is not None:
            return torch.addmm(self.bias, input_x, cat_kernels_4_quaternion)

        return torch.matmul(input_x, cat_kernels_4_quaternion)

    @staticmethod
    def weight_tensors(weight_shape):
        """Create and initialise the weight tensors according to quaternion rules."""
        modulus = nn.Parameter(torch.Tensor(*weight_shape))
        modulus = nn.init.xavier_uniform_(modulus, gain=1.0)

        i_weight = 2.0 * torch.rand(*weight_shape) - 1.0
        j_weight = 2.0 * torch.rand(*weight_shape) - 1.0
        k_weight = 2.0 * torch.rand(*weight_shape) - 1.0

        sum_imaginary_parts = i_weight.abs() + j_weight.abs() + k_weight.abs()

        i_weight = torch.div(i_weight, sum_imaginary_parts)
        j_weight = torch.div(j_weight, sum_imaginary_parts)
        k_weight = torch.div(k_weight, sum_imaginary_parts)

        phase = torch.rand(*weight_shape) * (2 * torch.tensor([np.pi])) - torch.tensor(
            [np.pi]
        )

        r_weight = modulus * np.cos(phase)
        i_weight = modulus * i_weight * np.sin(phase)
        j_weight = modulus * j_weight * np.sin(phase)
        k_weight = modulus * k_weight * np.sin(phase)

        return (
            nn.Parameter(r_weight),
            nn.Parameter(i_weight),
            nn.Parameter(j_weight),
            nn.Parameter(k_weight),
        )

    @staticmethod
    def get_weight_shape(in_channels, out_channels):
        """Construct weight shape based on the input/output channels."""
        return (in_channels, out_channels)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ")"
        )


class QuaternionConvolution(nn.Module):
    """Reproduction class of the quaternion convolution layer."""

    ALLOWED_DIMENSIONS = (2, 3)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dimension=2,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """Create the quaterion convolution layer."""
        super(QuaternionConvolution, self).__init__()

        self.in_channels = np.floor_divide(in_channels, 4) // groups
        self.out_channels = np.floor_divide(out_channels, 4)

        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.kernel_size = self.get_kernel_shape(kernel_size, dimension)
        self.weight_shape = self.get_weight_shape(
            self.in_channels, self.out_channels, self.kernel_size, self.groups
        )
        # print("weight shape", self.weight_shape)
        # print("in channels", self.in_channels)
        # print("out channels", self.out_channels)
        self._weights = self.weight_tensors(self.weight_shape)
        self.r_weight, self.k_weight, self.i_weight, self.j_weight = self._weights

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None

    def forward(self, x):
        """Apply forward pass of input through quaternion convolution layer."""
        cat_kernels_4_r = torch.cat(
            [self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=1
        )
        cat_kernels_4_i = torch.cat(
            [self.i_weight, self.r_weight, -self.k_weight, self.j_weight], dim=1
        )
        cat_kernels_4_j = torch.cat(
            [self.j_weight, self.k_weight, self.r_weight, -self.i_weight], dim=1
        )
        cat_kernels_4_k = torch.cat(
            [self.k_weight, -self.j_weight, self.i_weight, self.r_weight], dim=1
        )

        cat_kernels_4_quaternion = torch.cat(
            [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
        )

        if x.dim() == 3:
            convfunc = F.conv1d
        elif x.dim() == 4:
            convfunc = F.conv2d
        elif x.dim() == 5:
            convfunc = F.conv3d
        else:
            raise InvalidInput("Given input channels do not match allowed dimensions")
        # print(cat_kernels_4_quaternion.shape)
        return convfunc(
            x,
            cat_kernels_4_quaternion,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @staticmethod
    def weight_tensors(weight_shape):
        """Create and initialise the weight tensors according to quaternion rules."""
        modulus = nn.Parameter(torch.Tensor(*weight_shape))
        modulus = nn.init.xavier_uniform_(modulus, gain=1.0)

        i_weight = 2.0 * torch.rand(*weight_shape) - 1.0
        j_weight = 2.0 * torch.rand(*weight_shape) - 1.0
        k_weight = 2.0 * torch.rand(*weight_shape) - 1.0

        sum_imaginary_parts = i_weight.abs() + j_weight.abs() + k_weight.abs()

        i_weight = torch.div(i_weight, sum_imaginary_parts)
        j_weight = torch.div(j_weight, sum_imaginary_parts)
        k_weight = torch.div(k_weight, sum_imaginary_parts)

        phase = torch.rand(*weight_shape) * (2 * torch.tensor([np.pi])) - torch.tensor(
            [np.pi]
        )

        r_weight = modulus * np.cos(phase)
        i_weight = modulus * i_weight * np.sin(phase)
        j_weight = modulus * j_weight * np.sin(phase)
        k_weight = modulus * k_weight * np.sin(phase)

        return (
            nn.Parameter(r_weight),
            nn.Parameter(i_weight),
            nn.Parameter(j_weight),
            nn.Parameter(k_weight),
        )

    @staticmethod
    def get_weight_shape(in_channels, out_channels, kernel_size, groups):
        """Construct weight shape based on the input/output channels and kernel size."""
        return (out_channels, in_channels) + kernel_size

    @staticmethod
    def get_kernel_shape(kernel_size, dimension):
        """Construct the kernel shape based on the given dimension and kernel size."""
        if dimension not in QuaternionConvolution.ALLOWED_DIMENSIONS:
            raise InvalidKernelShape("Given dimensions are not allowed.")

        if isinstance(kernel_size, int):
            return (kernel_size,) * dimension

        if isinstance(kernel_size, tuple):
            if len(kernel_size) != dimension:
                raise InvalidKernelShape("Given kernel shape does not match dimension.")

            return kernel_size

        raise InvalidKernelShape("No valid type of kernel size to construct kernel.")

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels * 4)
            + ", out_channels="
            + str(self.out_channels * 4)
            + ", kernel_size="
            + str(self.kernel_size)
            + ", stride="
            + str(self.stride)
            + ", padding="
            + str(self.padding)
            + ", dilation="
            + str(self.dilation)
            + ", groups="
            + str(self.groups)
            + ")"
        )


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        groups_per_layer = [0 for _ in range(2)]
        if num_groups != 0:
            groups_per_layer[0] = (
                in_channels // num_groups
                if in_channels <= mid_channels
                else mid_channels // num_groups
            )
            groups_per_layer[1] = (
                mid_channels // num_groups
                if mid_channels <= out_channels
                else out_channels // num_groups
            )

        self.double_conv = nn.Sequential(
            QuaternionConvolution(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                bias=False,
                groups=groups_per_layer[0] if groups_per_layer[0] != 0 else 1,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            QuaternionConvolution(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                bias=False,
                groups=groups_per_layer[1] if groups_per_layer[1] != 0 else 1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, num_groups=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, num_groups=num_groups),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        num_groups=0,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, num_groups=num_groups
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
                groups=(in_channels // 2) // num_groups if num_groups != 0 else 1,
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                num_groups=num_groups,
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
