class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_last_val(self):
        return self.val


class ExpAverage(object):
    def __init__(self, alpha, init_val=0):
        self.val = init_val
        self.avg = init_val
        self.alpha = alpha

    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.avg + (1 - self.alpha) * val

    def get_avg(self):
        return self.avg

    def get_last_val(self):
        return self.val
