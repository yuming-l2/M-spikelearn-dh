import torchvision
trainset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data', train=True, download=True)
testset=torchvision.datasets.CIFAR10(root='/tmp/torchvision_data', train=False, download=True)
