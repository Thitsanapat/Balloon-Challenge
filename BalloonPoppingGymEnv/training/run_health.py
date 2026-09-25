"""Training-run health policy, independent of simulator and learned policy."""


class TruncationGuard:
    def __init__(self, limit=3):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError('Positive integer truncation limit required')
        self.limit = limit
        self.consecutive = 0
        self.total = 0

    def observe(self, truncated):
        self.total += int(bool(truncated))
        self.consecutive = self.consecutive+1 if truncated else 0
        return self.consecutive >= self.limit
