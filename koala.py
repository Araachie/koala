import torch

from util import ExpAverage


class KOALABase(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        defaults = dict(**kwargs)
        super(KOALABase, self).__init__(params, defaults)

    @torch.no_grad()
    def predict(self):
        pass

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        pass


class VanillaKOALA(KOALABase):
    def __init__(
            self,
            params,
            sigma: float = 1,
            q: float = 1,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1,
            **kwargs):
        """
        Implementation of the KOALA-V(Vanilla) optimizer

        :param params: parameters to optimize
        :param sigma: initial value of P_k
        :param q: fixed constant Q_k
        :param r: fixed constant R_k (None for online estimation)
        :param alpha_r: smoothing coefficient for online estimation of R_k
        :param weight_decay: weight decay
        :param lr: learning rate
        :param kwargs:
        """
        super(VanillaKOALA, self).__init__(params, **kwargs)

        self.eps = 1e-9

        for group in self.param_groups:
            group["lr"] = lr

        # Initialize state
        self.state["sigma"] = sigma
        self.state["q"] = q
        if r is not None:
            self.state["r"] = r
        else:
            self.state["r"] = ExpAverage(alpha_r, 1.0)
        self.state["weight_decay"] = weight_decay

    @torch.no_grad()
    def predict(self):
        self.state["sigma"] += self.state["q"]

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        if isinstance(self.state["r"], ExpAverage):
            self.state["r"].update(loss_var)
            cur_r = self.state["r"].get_avg()
        else:
            cur_r = self.state["r"]

        max_grad_entries = list()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.norm(p=2) < self.eps:
                    continue

                layer_grad = p.grad + self.state["weight_decay"] * p
                layer_grad_norm = layer_grad.norm(p=2)

                s = self.state["sigma"] * (layer_grad_norm ** 2) + cur_r

                layer_loss = loss + 0.5 * self.state["weight_decay"] * p.norm(p=2) ** 2
                scale = group["lr"] * layer_loss * self.state["sigma"] / s
                p.data.add_(-scale * p.grad)

                max_grad_entries.append(layer_grad_norm ** 2 / s)

        hh_approx = torch.max(torch.stack(max_grad_entries))

        self.state["sigma"] -= self.state["sigma"] ** 2 * hh_approx


class MomentumKOALA(KOALABase):
    def __init__(
            self,
            params,
            sw: float = 1e-1,
            sc: float = 0,
            sv: float = 1e-1,
            a: float = 0.9,
            qw: float = 1e-2,
            qv: float = 1e-2,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1,
            **kwargs):
        """
        Implementation of the KOALA-M(Momentum) optimizer

        :param params: parameters to optimize
        :param sw: initial value of P_k for states
        :param sc: initial value of out of diagonal entries of P_k
        :param sv: initial value of P_k for velocities
        :param a: decay coefficient for velocities
        :param qw: fixed constant Q_k for states
        :param qv: fixed constant Q_k for velocities
        :param r: fixed constant R_k (None for online estimation)
        :param alpha_r: smoothing coefficient for online estimation of R_k
        :param weight_decay: weight decay
        :param lr: learning rate
        :param kwargs:
        """
        super(MomentumKOALA, self).__init__(params, **kwargs)

        self.eps = 1e-9

        self.shared_device = self.param_groups[0]["params"][0].device
        self.dtype = torch.double

        # Initialize velocities and count params
        self.total_params = 0
        for group in self.param_groups:
            group["lr"] = lr
            for p in group["params"]:
                self.state[p]["vt"] = p.new_zeros(p.shape)
                self.state[p]["gt"] = p.new_zeros(p.shape)
                self.total_params += torch.prod(torch.Tensor(list(p.size())).to(self.shared_device))

        # Define state
        self.state["Pt"] = torch.Tensor([
            [sw, sc],
            [sc, sv]
        ]).to(self.shared_device).to(self.dtype)

        self.state["qw"] = ExpAverage(0.9, qw)
        self.state["qv"] = qv
        self.state["Q"] = torch.diag(
            torch.Tensor([self.state["qw"].get_avg(), self.state["qv"]])
        ).to(self.shared_device).to(self.dtype)

        if r is not None:
            self.state["R"] = r
        else:
            self.state["R"] = ExpAverage(alpha_r, 1.0)

        f = [[1, 1], [0, a]]
        self.state["F"] = torch.Tensor(f).to(self.shared_device).to(self.dtype)

        self.state["weight_decay"] = weight_decay

    @torch.no_grad()
    def predict(self):
        wdiff = list()
        for group in self.param_groups:
            for p in group["params"]:
                pw_diff = (self.state[p]["gt"] - p).norm(p=2).to(self.shared_device)
                wdiff.append(pw_diff)

                p.mul_(self.state["F"][0, 0].to(p.device))
                p.add_(self.state[p]["vt"] * self.state["F"][0, 1].to(p.device))
                self.state[p]["vt"].mul_(self.state["F"][1, 1].to(p.device))
                self.state[p]["vt"].add_(p * self.state["F"][1, 0].to(p.device))

        norm_wdiff = torch.stack(wdiff).norm(p=2) / self.total_params
        self.state["qw"].update(norm_wdiff)
        self.state["Q"] = torch.diag(
            torch.Tensor([self.state["qw"].get_avg(), self.state["qv"]])
        ).to(self.shared_device).to(self.dtype)

        self.state["Pt"] = torch.matmul(
            torch.matmul(self.state["F"], self.state["Pt"]), self.state["F"].t())
        self.state["Pt"].add_(self.state["Q"])

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        if isinstance(self.state["R"], ExpAverage):
            self.state["R"].update(loss_var.to(self.shared_device))
            cur_r = self.state["R"].get_avg()
        else:
            cur_r = self.state["R"]

        max_grad_entries = list()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.norm(p=2) < self.eps:
                    continue

                layer_grad = p.grad + self.state["weight_decay"] * p
                layer_grad_norm = layer_grad.norm(p=2)

                S = layer_grad_norm ** 2 * self.state["Pt"][0, 0] + cur_r

                layer_loss = loss.to(self.shared_device) + 0.5 * self.state["weight_decay"] * p.norm(p=2) ** 2
                K1 = self.state["Pt"][0, 0] / S * layer_loss * group["lr"]
                K2 = self.state["Pt"][1, 0] / S * layer_loss * group["lr"]

                # Update weights and velocities
                p.sub_((K1 * layer_grad).to(p.device))
                self.state[p]["vt"].sub_((K2 * layer_grad).to(p.device))

                self.state[p]["gt"].mul_(0.9)
                self.state[p]["gt"].add_(0.1 * p)

                max_grad_entries.append(layer_grad_norm ** 2 / S)

        hh_approx = torch.max(torch.stack(max_grad_entries))

        # Update covariance
        HHS = torch.Tensor([
            [hh_approx, 0],
            [0, 0]
        ]).to(self.shared_device).to(self.dtype)
        PHHS = torch.matmul(self.state["Pt"], HHS)
        PHHSP = torch.matmul(PHHS, self.state["Pt"].t())
        self.state["Pt"] = self.state["Pt"] - PHHSP
