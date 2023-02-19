from torchvision.datasets import CelebA
import torchvision.transforms as T 

dataset = CelebA(
    root="../../data", split = 'train', target_type = 'identity', 
    download = True, transform=T.Compose([
        T.Resize((196, 196)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
print(dataset[0][0].shape)
print(dataset[1][0].shape)
print(dataset[2][0].shape)
# dataset[1][0].save('sample.jpg')
# for img, tar in dataset[:10]:
#     print(img.shape, tar)