import copy
import torch
from landmarker.heatmap.generator import GaussianHeatmapGenerator
from landmarker.schedulers.adaloss import AdalossScheduler


def test_adaloss_scheduler():
    # create an instance of the scheduler
    scheduler = AdalossScheduler(nb_landmarks=5, rho=0.9, window=3, non_increasing=False)

    heatmap_generator = GaussianHeatmapGenerator(nb_landmarks=5, sigmas=10)

    losses = torch.tensor([0.2, 0.1, 0.2, 0.1, 0.2])
    # call the scheduler with the input losses and sigmas
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length
    for i in range(5):
        assert len(scheduler.losses[i]) == 1

    # check that the prev_variances list has the correct length
    assert len(scheduler.prev_variances) == 0

    losses = losses*0.9

    # call the scheduler with the same input losses and sigmas again
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length
    for i in range(5):
        assert len(scheduler.losses[i]) == 2

    # check that the prev_variances list has the correct length
    assert len(scheduler.prev_variances) == 0

    losses = losses*0.9

    # call the scheduler with the same input losses and sigmas again
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length
    for i in range(5):
        assert len(scheduler.losses[i]) == 3

    # check that the prev_variances list has the correct length
    assert len(scheduler.prev_variances) != 0

    losses = losses*0.9

    # call the scheduler with the same input losses and sigmas again
    prev_sigmas = copy.deepcopy(heatmap_generator.sigmas)
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length and that the prev variances are not zero
    for i in range(5):
        assert len(scheduler.losses[i]) == 3
        assert scheduler.prev_variances[i] != 0

    # check that the sigmas have been updated (they should be smaller)
    assert heatmap_generator.sigmas.shape == prev_sigmas.shape
    for i in range(5):
        assert heatmap_generator.sigmas[i, 0] < prev_sigmas[i, 0]
        assert heatmap_generator.sigmas[i, 1] < prev_sigmas[i, 1]


def test_adaloss_scheduler_non_increasing():
    # create an instance of the scheduler
    scheduler = AdalossScheduler(nb_landmarks=5, rho=0.9, window=3, non_increasing=True)

    heatmap_generator = GaussianHeatmapGenerator(nb_landmarks=5, sigmas=10)

    losses = torch.tensor([0.2, 0.1, 0.2, 0.1, 0.2])
    # call the scheduler with the input losses and sigmas
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length
    for i in range(5):
        assert len(scheduler.losses[i]) == 1

    # check that the prev_variances list has the correct length
    assert len(scheduler.prev_variances) == 0

    losses = losses*1.1

    # call the scheduler with the same input losses and sigmas again
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length
    for i in range(5):
        assert len(scheduler.losses[i]) == 2

    # check that the prev_variances list has the correct length
    assert len(scheduler.prev_variances) == 0

    losses = losses*1.1

    # call the scheduler with the same input losses and sigmas again
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length
    for i in range(5):
        assert len(scheduler.losses[i]) == 3

    # check that the prev_variances list has the correct length
    assert len(scheduler.prev_variances) != 0

    losses = losses*1.1

    # call the scheduler with the same input losses and sigmas again
    prev_sigmas = copy.deepcopy(heatmap_generator.sigmas)
    scheduler(losses, heatmap_generator.sigmas)

    # check that the losses deque has the correct length and that the prev variances are not zero
    for i in range(5):
        assert len(scheduler.losses[i]) == 3
        assert scheduler.prev_variances[i] != 0

    # check that the sigmas have been updated (they should be smaller)
    assert heatmap_generator.sigmas.shape == prev_sigmas.shape
    for i in range(5):
        assert heatmap_generator.sigmas[i, 0] == prev_sigmas[i, 0]
        assert heatmap_generator.sigmas[i, 1] == prev_sigmas[i, 1]
