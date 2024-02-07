"""Test landmark transforms."""

import pytest
import torch
from src.landmarker.transforms.landmarks import affine_landmarks, resize_landmarks


def test_resize_landmarks_without_padding():
    # Create a tensor of landmarks
    landmarks = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # Original and new dimensions
    orig_dim = (6, 6)
    new_dim = (12, 12)

    # Call the function
    resized_landmarks = resize_landmarks(landmarks, orig_dim, new_dim)

    # Check that the output has the expected values
    expected_output = torch.tensor([[2, 4], [6, 8], [10, 12]])
    assert torch.all(resized_landmarks.eq(expected_output))

    # Create a multi-instance tensor of landmarks
    landmarks = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                              [[7, 8], [9, 10], [11, 12]]])

    # Call the function
    resized_landmarks = resize_landmarks(landmarks, orig_dim, new_dim)

    # Check that the output has the expected values
    expected_output = torch.tensor([[[2, 4], [6, 8], [10, 12]],
                                    [[14, 16], [18, 20], [22, 24]]])
    assert torch.all(resized_landmarks.eq(expected_output))


def test_resize_landmarks_with_padding():
    # Original and new dimensions
    orig_dim = (40, 20)
    new_dim = (10, 10)
    padding = (0, 10)

    # Create a tensor of landmarks
    landmarks = torch.Tensor([[10, 20], [30, 10], [20, 5]])

    # Call the function
    resized_landmarks = resize_landmarks(landmarks, orig_dim, new_dim, padding)

    # Check that the output has the expected values
    expected_output = torch.Tensor([[2.5, 30.0 / 4], [30.0 / 4, 5], [5, 15 / 4]])
    assert torch.allclose(resized_landmarks, expected_output)

    # Create a multi-instance tensor of landmarks
    landmarks = torch.Tensor([[[10, 20], [30, 10], [20, 5]],
                              [[20, 10], [10, 30], [5, 20]]])

    # Call the function
    resized_landmarks = resize_landmarks(landmarks, orig_dim, new_dim, padding)

    # Check that the output has the expected values
    expected_output = torch.Tensor([[[2.5, 30.0 / 4], [30.0 / 4, 5], [5, 15 / 4]],
                                    [[5, 5], [2.5, 10], [5 / 4, 30 / 4]]])
    assert torch.allclose(resized_landmarks, expected_output)


def test_affine_landmarks_2d():
    # Create a tensor of 2D landmarks
    landmarks = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # Create an affine matrix
    affine_matrix = torch.eye(4)

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = landmarks
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (shift by 1)
    landmarks = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[1, 0, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[2, 3], [4, 5], [6, 7]])
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (90 degrees rotation)
    landmarks = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[0, -1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[-2, 1], [-4, 3], [-6, 5]])
    assert torch.all(transformed_landmarks.eq(expected_output))


def test_affine_landmarks_2d_multi_instance():
    # Create a multi-instance tensor of 2D landmarks
    landmarks = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                              [[7, 8], [9, 10], [11, 12]]])

    # Create an affine matrix
    affine_matrix = torch.eye(4)

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = landmarks
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (shift by 1)
    landmarks = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                              [[7, 8], [9, 10], [11, 12]]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[1, 0, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[[2, 3], [4, 5], [6, 7]],
                                    [[8, 9], [10, 11], [12, 13]]])
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (90 degrees rotation)
    landmarks = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                              [[7, 8], [9, 10], [11, 12]]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[0, -1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[[[-2, 1], [-4, 3], [-6, 5]],
                                    [[-8, 7], [-10, 9], [-12, 11]]]])
    assert torch.all(transformed_landmarks.eq(expected_output))


def test_affine_landmarks_3d():
    # Create a tensor of 3D landmarks
    landmarks = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Create an affine matrix
    affine_matrix = torch.eye(4)

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = landmarks
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (shift by 1)
    landmarks = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[1, 0, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (90 degrees rotation)
    landmarks = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[0, -1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[-2, 1, 3], [-5, 4, 6], [-8, 7, 9]])
    assert torch.all(transformed_landmarks.eq(expected_output))


def test_affine_landmarks_3d_multi_instance():
    # Create a multi-instance tensor of 3D landmarks
    landmarks = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                              [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])

    # Create an affine matrix
    affine_matrix = torch.eye(4)

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = landmarks
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (shift by 1)
    landmarks = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                              [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[1, 0, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[[2, 3, 4], [5, 6, 7], [8, 9, 10]],
                                    [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
    assert torch.all(transformed_landmarks.eq(expected_output))

    # Test with a non-identity affine matrix (90 degrees rotation)
    landmarks = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                              [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])

    # Create an affine matrix
    affine_matrix = torch.Tensor([[0, -1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Call the function
    transformed_landmarks = affine_landmarks(landmarks, affine_matrix)

    # Check that the output has the expected values
    expected_output = torch.tensor([[[-2, 1, 3], [-5, 4, 6], [-8, 7, 9]],
                                    [[-11, 10, 12], [-14, 13, 15], [-17, 16, 18]]])
    assert torch.all(transformed_landmarks.eq(expected_output))


def test_affine_landmarks_invalid():
    # Create a tensor of landmarks with invalid shape
    landmarks = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    # Create an affine matrix
    affine_matrix = torch.eye(4)

    # Call the function and check that it raises a ValueError
    with pytest.raises(ValueError):
        affine_landmarks(landmarks, affine_matrix)
