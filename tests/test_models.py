"""Tests for the models module."""

import os
from functools import partial

import torch
from monai.networks.blocks import ResidualUnit

from landmarker.models.cholesky_hourglass import StackedCholeskyHourglass
from landmarker.models.coord_conv import CoordConvLayer
from landmarker.models.hourglass import Hourglass, StackedHourglass
from landmarker.models.spatial_configuration_net import (
    OriginalSpatialConfigurationNet,
    OriginalSpatialConfigurationNet3d,
    ProbSpatialConfigurationNet,
    SpatialConfigurationNet,
)
from landmarker.models.utils import LogSoftmaxND, SoftmaxND


def test_original_spatial_configuration_net():
    """Test OriginalSpatialConfigurationNet."""
    net = OriginalSpatialConfigurationNet(in_channels=1, out_channels=4)
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = OriginalSpatialConfigurationNet(in_channels=1, out_channels=4, init_weigths=True)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape


def test_original_spatial_configuration_net_3d():
    """Test OriginalSpatialConfigurationNet3d."""
    net = OriginalSpatialConfigurationNet3d(in_channels=1, out_channels=4)
    input_tensor = torch.randn((1, 1, 192, 96, 96))
    expected_output_shape = (1, 4, 192, 96, 96)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape


def test_spatial_configuration_net():
    """Test SpatialConfigurationNet."""
    net = SpatialConfigurationNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        la_channels=(128, 128, 128, 128),
        la_strides=(2, 2, 2),
        la_num_res_units=2,
        la_norm="instance",
        la_dropout=0.0,
        sp_channels=128,
        sp_kernel_size=11,
        sp_downsample=16,
    )
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = SpatialConfigurationNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        la_channels=(128, 128, 128, 128),
        la_strides=(2, 2, 2),
        la_num_res_units=2,
        la_norm="instance",
        la_dropout=0.0,
        sp_channels=128,
        sp_kernel_size=11,
        sp_downsample=16,
        sp_image_input=False,
    )

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape


def test_prob_spatial_configuration_net():
    """Test ProbSpatialConfigurationNet."""
    net = ProbSpatialConfigurationNet(spatial_dims=2, in_channels=1, out_channels=4)
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = ProbSpatialConfigurationNet(
        spatial_dims=2, in_channels=1, out_channels=4, sp_image_input=True
    )

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = ProbSpatialConfigurationNet(
        spatial_dims=2, in_channels=1, out_channels=4, sp_image_input=False
    )

    assert output_tensor.shape == expected_output_shape


def test_hourglass():
    """Test Hourglass.""" ""
    spatial_dims = 2
    in_channels = 1
    out_channels = 4
    channels = [16, 32, 64, 64]
    subunits = 3
    up_sample_mode = "nearest"

    model = Hourglass(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        conv_block=partial(ResidualUnit, norm="batch", subunits=subunits),
        up_sample_mode=up_sample_mode,
    )
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, out_channels, 256, 256)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == expected_output_shape


def test_stacked_hourglass():
    """Test StackedHourglass."""
    nb_stacks = 2
    spatial_dims = 2
    in_channels = 3
    out_channels = 4
    channels = [16, 32, 64, 64]
    subunits = 3
    up_sample_mode = "nearest"

    model = StackedHourglass(
        nb_stacks=nb_stacks,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        conv_block=partial(ResidualUnit, norm="batch", subunits=subunits),
        up_sample_mode=up_sample_mode,
    )
    input_tensor = torch.randn((1, 3, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = model(input_tensor)

    assert len(output_tensor) == 2
    for i in range(nb_stacks):
        assert output_tensor[i].shape == expected_output_shape


def test_stack_cholesky_hourglass():
    """
    Test StackedCholeskyHourglass.
    """
    nb_stacks = 2
    img_size = (256, 256)
    in_channels = 1
    out_channels = 4
    channels = [16, 32, 64, 64]
    subunits = 3
    up_sample_mode = "nearest"

    model = StackedCholeskyHourglass(
        nb_stacks=nb_stacks,
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        conv_block=partial(ResidualUnit, norm="batch", subunits=subunits),
        up_sample_mode=up_sample_mode,
    )
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_hm_output_shape = (1, 4, 256, 256)
    expected_cholesky_output_shape = (1, 4, 2, 2)

    hm_output_tensor, cholesky_output_tensor = model(input_tensor)

    assert len(hm_output_tensor) == nb_stacks
    assert len(cholesky_output_tensor) == nb_stacks
    assert hm_output_tensor[0].shape == expected_hm_output_shape
    assert cholesky_output_tensor[0].shape == expected_cholesky_output_shape


def test_coord_conv_layer_no_radial_channel():
    """
    Test CoordConvLayer without radial channel.
    """
    # create an instance of the CoordConvLayer class without radial channel
    layer = CoordConvLayer(
        spatial_dims=2,
        in_channels=3,
        out_channels=16,
        radial_channel=False,
        conv_block=partial(ResidualUnit, strides=4, kernel_size=7),
    )

    # create some dummy input tensors
    x = torch.randn(2, 3, 64, 64)

    # pass the input tensor through the layer
    out = layer(x)

    # check that the output has the correct shape
    assert out.shape == torch.Size([2, 16, 16, 16])


def test_coord_conv_layer_with_radial_channel():
    """
    Test CoordConvLayer with radial channel.
    """
    # create an instance of the CoordConvLayer class with radial channel
    layer = CoordConvLayer(
        spatial_dims=2,
        in_channels=3,
        out_channels=16,
        radial_channel=True,
        conv_block=partial(ResidualUnit, strides=1, kernel_size=3),
    )

    # create some dummy input tensors
    x = torch.randn(2, 3, 64, 64)

    # pass the input tensor through the layer
    out = layer(x)

    # check that the output has the correct shape
    assert out.shape == torch.Size([2, 16, 64, 64])


def test_coord_conv_layer_coord_channels_range():
    """
    Test that the values of the coordinate channels are within the range [-1, 1].
    """
    # create an instance of the CoordConvLayer class without radial channel
    layer = CoordConvLayer(
        spatial_dims=2,
        in_channels=3,
        out_channels=16,
        radial_channel=False,
        conv_block=partial(ResidualUnit, strides=1, kernel_size=3),
    )

    # create some dummy input tensors
    x = torch.randn(2, 3, 64, 64)

    # pass the input tensor through the layer
    out = layer.add_coord_channels(x)

    # check that the output values are within the range [-1, 1]
    assert out.shape == torch.Size([2, 5, 64, 64])
    assert (-1 <= out[:, 3:]).all() and (out[:, 3:] <= 1).all()


def test_softmax_nd():
    """Test the SoftmaxND class."""
    # Test for 2D case
    softmax_2d = SoftmaxND(spatial_dims=2)
    x = torch.randn(1, 3, 4, 4)
    output = softmax_2d(x)
    assert output.shape == x.shape
    assert torch.allclose(torch.sum(output, dim=(-2, -1)), torch.ones(1, 3))

    # Test for 3D case
    softmax_3d = SoftmaxND(spatial_dims=3)
    x = torch.randn(1, 3, 4, 4, 4)
    output = softmax_3d(x)
    assert output.shape == x.shape
    assert torch.allclose(torch.sum(output, dim=(-3, -2, -1)), torch.ones(1, 3))


def test_log_softmax_nd():
    """Test the LogSoftmaxND class."""
    # Test for 2D case
    log_softmax_2d = LogSoftmaxND(spatial_dims=2)
    x = torch.randn(1, 3, 4, 4)
    output = log_softmax_2d(x)
    assert output.shape == x.shape
    assert torch.allclose(torch.sum(torch.exp(output), dim=(-2, -1)), torch.ones(1, 3))

    # Test for 3D case
    log_softmax_3d = LogSoftmaxND(spatial_dims=3)
    x = torch.randn(1, 3, 4, 4, 4)
    output = log_softmax_3d(x)
    assert output.shape == x.shape
    assert torch.allclose(torch.sum(torch.exp(output), dim=(-3, -2, -1)), torch.ones(1, 3))
