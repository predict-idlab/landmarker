from landmarker.uncertainty import (  # type: ignore
    MR2C2R,
    MR2CCP,
    ConformalRegressorBonferroni,
    ConformalRegressorMahalanobis,
    ConformalRegressorMaxNonconformity,
    ContourHuggingRegressor,
    MultivariateNormalRegressor,
    resize_landmarks,
    transform_heatmap_to_original_size,
    transform_heatmap_to_original_size_numpy,
)


def test_import_uncertainty():
    """Test that the uncertainty module can be imported correctly."""
    assert MR2C2R is not None
    assert MR2CCP is not None
    assert ConformalRegressorBonferroni is not None
    assert ConformalRegressorMahalanobis is not None
    assert ConformalRegressorMaxNonconformity is not None
    assert ContourHuggingRegressor is not None
    assert MultivariateNormalRegressor is not None
    assert resize_landmarks is not None
    assert transform_heatmap_to_original_size is not None
    assert transform_heatmap_to_original_size_numpy is not None
