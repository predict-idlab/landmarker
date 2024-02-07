"""Tests for the models module."""

from functools import partial
import os

import torch
from monai.networks.blocks import ResidualUnit

from landmarker.models.spatial_configuration_net import (SpatialConfigurationNet,
                                                         ProbSpatialConfigurationNet,
                                                         OriginalSpatialConfigurationNet)
from landmarker.models import get_model
from landmarker.models.hourglass import Hourglass, StackedHourglass
from landmarker.models.cholesky_hourglass import StackedCholeskyHourglass
from landmarker.models.coord_conv import CoordConvLayer


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


def test_spatial_configuration_net():
    """Test SpatialConfigurationNet."""
    net = SpatialConfigurationNet(spatial_dims=2, in_channels=1, out_channels=4,
                                  la_channels=(128, 128, 128, 128),
                                  la_strides=(2, 2, 2),
                                  la_num_res_units=2,
                                  la_norm="instance",
                                  la_dropout=0.0,
                                  sp_channels=128,
                                  sp_kernel_size=11,
                                  sp_downsample=16)
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = SpatialConfigurationNet(spatial_dims=2, in_channels=1, out_channels=4,
                                  la_channels=(128, 128, 128, 128),
                                  la_strides=(2, 2, 2),
                                  la_num_res_units=2,
                                  la_norm="instance",
                                  la_dropout=0.0,
                                  sp_channels=128,
                                  sp_kernel_size=11,
                                  sp_downsample=16,
                                  sp_image_input=False)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape


def test_prob_spatial_configuration_net():
    """Test ProbSpatialConfigurationNet."""
    net = ProbSpatialConfigurationNet(spatial_dims=2, in_channels=1, out_channels=4)
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = ProbSpatialConfigurationNet(spatial_dims=2, in_channels=1, out_channels=4,
                                      sp_image_input=False)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape


def test_hourglass():
    """Test Hourglass."""""
    spatial_dims = 2
    in_channels = 1
    out_channels = 4
    channels = [16, 32, 64, 64]
    subunits = 3
    up_sample_mode = 'nearest'

    model = Hourglass(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                      channels=channels, conv_block=partial(ResidualUnit, norm="batch",
                                                            subunits=subunits),
                      up_sample_mode=up_sample_mode)
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
    up_sample_mode = 'nearest'

    model = StackedHourglass(nb_stacks=nb_stacks, spatial_dims=spatial_dims,
                             in_channels=in_channels, out_channels=out_channels, channels=channels,
                             conv_block=partial(ResidualUnit, norm="batch", subunits=subunits),
                             up_sample_mode=up_sample_mode)
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
    up_sample_mode = 'nearest'

    model = StackedCholeskyHourglass(nb_stacks=nb_stacks, img_size=img_size,
                                     in_channels=in_channels, out_channels=out_channels,
                                     channels=channels, conv_block=partial(
                                         ResidualUnit, norm="batch", subunits=subunits),
                                     up_sample_mode=up_sample_mode)
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
    layer = CoordConvLayer(spatial_dims=2, in_channels=3, out_channels=16, radial_channel=False,
                           conv_block=partial(ResidualUnit, strides=4, kernel_size=7))

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
    layer = CoordConvLayer(spatial_dims=2, in_channels=3, out_channels=16, radial_channel=True,
                           conv_block=partial(ResidualUnit, strides=1, kernel_size=3))

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
    layer = CoordConvLayer(spatial_dims=2, in_channels=3, out_channels=16, radial_channel=False,
                           conv_block=partial(ResidualUnit, strides=1, kernel_size=3))

    # create some dummy input tensors
    x = torch.randn(2, 3, 64, 64)

    # pass the input tensor through the layer
    out = layer.add_coord_channels(x)

    # check that the output values are within the range [-1, 1]
    assert out.shape == torch.Size([2, 5, 64, 64])
    assert (-1 <= out[:, 3:]).all() and (out[:, 3:] <= 1).all()


def test_get_model():
    """Test get_model."""
    net = get_model("SpatialConfigurationNet", spatial_dims=2, in_channels=1, out_channels=4,
                    la_channels=(128, 128, 128, 128),
                    la_strides=(2, 2, 2),
                    la_num_res_units=2,
                    la_norm="instance",
                    la_dropout=0.0,
                    sp_channels=128,
                    sp_kernel_size=11,
                    sp_downsample=16)
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("OriginalSpatialConfigurationNet",
                    spatial_dims=2, in_channels=1, out_channels=4)
    input_tensor = torch.randn((1, 1, 256, 256))
    expected_output_shape = (1, 4, 256, 256)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("ProbSpatialConfigurationNet", spatial_dims=2, in_channels=1, out_channels=4)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("CoordConvProbSpatialConfigurationNet", spatial_dims=2, in_channels=1,
                    out_channels=4)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("UNet", spatial_dims=2, in_channels=1, out_channels=4)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("UNet", spatial_dims=2, in_channels=1, out_channels=4, channels=(4, 8, 16),
                    strides=(2, 2))

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("CoordConvUNet", spatial_dims=2, in_channels=1, out_channels=4)

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    net = get_model("CoordConvUNet", spatial_dims=2, in_channels=1, out_channels=4,
                    channels=(4, 8, 16), strides=(2, 2))

    output_tensor = net(input_tensor)

    assert output_tensor.shape == expected_output_shape

    try:
        net = get_model("unvalid-name")
        assert False
    except ValueError:
        assert True


def test_get_model_load():
    """Test load saved model functionality with get_model()"""
    net = get_model("SpatialConfigurationNet", spatial_dims=2, in_channels=1, out_channels=4,
                    la_channels=(128, 128, 128, 128),
                    la_strides=(2, 2, 2),
                    la_num_res_units=2,
                    la_norm="instance",
                    la_dropout=0.0,
                    sp_channels=128,
                    sp_kernel_size=11,
                    sp_downsample=16)

    torch.save(net.state_dict(), "tests/scn_test.pt")
    input_tensor = torch.randn((1, 1, 256, 256))

    output_tensor = net(input_tensor)

    net_load = get_model("SpatialConfigurationNet", spatial_dims=2, in_channels=1, out_channels=4,
                         la_channels=(128, 128, 128, 128),
                         la_strides=(2, 2, 2),
                         la_num_res_units=2,
                         la_norm="instance",
                         la_dropout=0.0,
                         sp_channels=128,
                         sp_kernel_size=11,
                         sp_downsample=16,
                         path_to_model="tests/scn_test.pt")

    output_tensor_load = net_load(input_tensor)

    assert torch.allclose(output_tensor, output_tensor_load)

    os.remove("tests/scn_test.pt")
