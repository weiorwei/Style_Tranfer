import imageio
import torch
from torch.autograd import Variable
import torchvision
from torchvision.io import read_image
from PIL import Image
from torch import nn
from typing import Dict, Iterable, Callable
import numpy as np
from skimage.transform import resize

weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
model = torchvision.models.vgg19(weights=weights)
model.eval()

print(model)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        weight = torchvision.models.VGG19_Weights.IMAGENET1K_V1  # vgg19 模型参数
        self.model = torchvision.models.vgg19(weights=weight)  # 加载vgg19 模型


    def forward(self, x):
        x = self.model(x)
        return x

mine =Net()
# 读取图片
img = imageio.imread('men.jpg')
print("处理前图片形式")
print(img.shape)
# 对读入的图片按照模型要求的形式处理
img = resize(img, (int(img.shape[0] / 3), int(img.shape[1] / 3)), preserve_range=True)
img = torch.from_numpy(img)  # 将图片从numpy.ndarray转换成Tensor数据类型
img = np.transpose(img, (2, 0, 1))  # 将图片从（450，300，3）转换成（3，450，300）
img = img.unsqueeze(0).to(torch.float32)

print("处理后图片形式")
print(img.shape)
res = mine(img).squeeze(0).softmax(0).argmax().item()
res_m = mine(img).detach()
print(res_m)
print('输入')
print(img)
print('结果')
print("res", res)
print(img.shape)
print(mine)

def image_loader(image_name):
    image = Image.open(image_name)

    image = loader(image).unsqueeze(0)  # 需要伪造的批次尺寸以适合网络的输入尺寸
    return image.to(device, torch.float)


#
#
# class LayerActivations:
#     features = None
#
#     def __init__(self, model, layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output):
#         self.features = output.cpu()
#
#     def remove(self):
#         self.hook.remove()
#
# # 选取特定卷积层的输出
# pre = model.features[0](img)  # 这一步为 该模型的第一层输出结果
# print('conv_layer1')
# print(pre.shape)
# pre = model.features[1](pre)  # 这一步为 该模型的第二层输出结果
# pre = model.features[2](pre)  # 这一步为 该模型的第三层输出结果
# print('conv_layer2')
# print(pre.shape)
# pre = model.features[3](pre)  # 这一步为 该模型的第四层输出结果
# pre = model.features[4](pre)  # 这一步为 该模型的第五层输出结果
# pre = model.features[5](pre)  # 这一步为 该模型的第六层输出结果
# print('conv_layer3')
# print(pre.shape)
# pre = model.features[6](pre)
# pre = model.features[7](pre)
# print('conv_layer4')
# print(pre.shape)
# pre = model.features[8](pre)
# pre = model.features[9](pre)
# pre = model.features[10](pre)
# print('conv_layer5')
# print(pre.shape)
# pre = model.features[11](pre)
# pre = model.features[12](pre)
# pre = model.features[13](pre)
# pre = model.features[14](pre)
# pre = model.features[15](pre)
# pre = model.features[16](pre)
# pre = model.features[17](pre)
# pre = model.features[18](pre)
# pre = model.features[19](pre)
# pre = model.features[20](pre)
# pre = model.features[21](pre)
# pre = model.features[22](pre)
# pre = model.features[23](pre)
# pre = model.features[24](pre)
# pre = model.features[25](pre)
# pre = model.features[26](pre)
# pre = model.features[27](pre)
# pre = model.features[28](pre)
# pre = model.features[29](pre)
# pre = model.features[30](pre)
# pre = model.features[31](pre)
# pre = model.features[32](pre)
# pre = model.features[33](pre)
# pre = model.features[34](pre)
# pre = model.features[35](pre)
# pre = model.features[36](pre)
# def select(img,model):
#     pre=[]
#     pre.append(img)
#     for i in range(36):
#         pre.append(model.features[i](pre[i]))
#     return pre
# res_2=select(img,model)
# print(res_2[1].shape)
# x=res_2[1]
# print(x.shape[1])
# print(1)
# from torchvision.io import read_image
# from torchvision.models import resnet50, ResNet50_Weights
#
# img = read_image("men.jpg")
#
# # Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.eval()
#
# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()
#
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)
# print(batch.shape)
# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(class_id)
# print(f"{category_name}: {100 * score:.1f}%")
# print(1)
