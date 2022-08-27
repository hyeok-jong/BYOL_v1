import torch
from byol_pytorch import BYOL
from torchvision import models, datasets, transforms
from tqdm import tqdm


resnet = models.resnet50(weights = None).to('cuda:1')

learner = BYOL(
    resnet,
    image_size = 64,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)



train_dataset = datasets.CIFAR10(root='./datasets', transform = transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True,
        num_workers=4, pin_memory=True)


for _ in tqdm(range(50)):
    for images, labels in train_loader:
        loss = learner(images.to('cuda:1'))
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() 

torch.save(resnet.state_dict(), './improved-net.pt')

