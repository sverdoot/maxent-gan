import torch


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return ((x1 - x2) ** 2).sum().sqrt()


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1 ** k).mean(0)
    ss2 = (sx2 ** k).mean(0)
    return l2diff(ss1, ss2)


class CMD(object):
    def __init__(self, n_moments=5):
        self.n_moments = n_moments

    def __call__(self, x1, x2):
        mx1 = x1.mean(dim=0)
        mx2 = x2.mean(dim=0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = l2diff(mx1, mx2)
        scms = dm

        for i in range(self.n_moments - 1):
            # moment diff of centralized samples
            scms += moment_diff(sx1, sx2, i + 2)
        return scms

    def moments(self, x: torch.FloatTensor) -> torch.FloatTensor:
        moment_id = 1
        result = x  # (moment_id + 1) * ((moment_id + 1) / moment_id) ** moment_id * x

        for moment_id in range(2, self.n_moments + 1):
            result = torch.cat([result, (x - x.mean(0).detach()) ** moment_id], 1)
        return (
            result  # / ((moment_id + 1) * ((moment_id + 1) / moment_id) ** moment_id)
        )
