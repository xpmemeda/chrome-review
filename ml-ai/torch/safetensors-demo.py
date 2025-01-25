import torch
from safetensors.torch import load_file, save_file


class Net(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self._fc = torch.nn.Linear(feature_size, feature_size, True)

    def forward(self, x):
        return self._fc(x)


x = torch.rand(1024)

net = Net(1024)
net.eval()
net2 = Net(1024)
net2.eval()

assert not torch.allclose(net2(x), net(x))

safetensors_path = "1.safetensors"
save_file(net.state_dict(), safetensors_path)
net2.load_state_dict(load_file(safetensors_path))

assert torch.allclose(net2(x), net(x))
