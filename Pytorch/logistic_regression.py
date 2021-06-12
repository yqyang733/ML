import torchvision     # 主要用于处理图像和视频
train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download = True)
test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download = True)

train_set = torchvision.datasets.CIFAR10()
test_set = torchvision.datasets.CIFAR10()   # 设置与MNIST中的设置一样