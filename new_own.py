import os
import torch.nn.functional as F
import imageio
import torch
import torchvision
from skimage.transform import resize
from torch import nn
import numpy as np
import time
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision import utils

weight = torchvision.models.VGG19_Weights.IMAGENET1K_V1  # vgg19 模型参数
vgg19 = torchvision.models.vgg19(weights=weight).features.to('cuda')  # 加载vgg19 模型
vgg19.eval()
print(vgg19)

# 以下为可以调的一些初始化参数
style_img_path = 'style_pic/sleep_flower.jpg'
content_img_path = 'origin_pic/proof.jpg'
IMAGE_HIGHT = 600
IMAGE_WIDTH = 900
epoch = 3001
save_path = 'neural_style_transfer_own/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
STYLE_Weight_LAYERS = [(0.1, 1), (1.0, 5), (1.5, 10), (3.0, 19), (6.0, 34)]  # 选择计算gram矩阵的卷积层，以及对应的权重
Weight = [0.5, 1.0, 1.5, 3.0, 4.0]
STYLE_LAYER = [0, 2, 5, 7, 10]
CONTEN_LAYERS = 7
Weight_Content = 5
Weight_Style = 10e7
# ------------------------------

loader = transforms.Compose([
    transforms.Resize((IMAGE_HIGHT, IMAGE_WIDTH)),  # 缩放导入的图像
    transforms.ToTensor()])  # 将其转换为torch tensor


def load_image(path):
    image = Image.open(path)
    image = loader(image).unsqueeze(0)  # 需要伪造的批次尺寸以适合网络的输入尺寸
    return image.to('cuda', torch.float)


# 计算图片均值
def aver_img(image):
    # 计算图片的平均值
    MEAN_VALUES = image.mean(axis=0)
    # MEAN_G = np.mean(image[:, :, 1])
    # MEAN_B = np.mean(image[:, :, 2])
    # MEAN_VALUES = torch.ones((image.shape[0], image.shape[1], image.shape[2]))
    # MEAN_VALUES[:, :, 0] = MEAN_VALUES[:, :, 0] * MEAN_R
    # MEAN_VALUES[:, :, 1] = MEAN_VALUES[:, :, 1] * MEAN_G
    # MEAN_VALUES[:, :, 2] = MEAN_VALUES[:, :, 2] * MEAN_B
    return MEAN_VALUES


# 打印时间
def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))


# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
#
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')
cnn_normalization_mean = torch.tensor([0.5, 0.5, 0.5]).to('cuda')

cnn_normalization_std = torch.tensor([0.5, 0.5, 0.5]).to('cuda')

class empty(nn.Module):
    # def __init__(self, image):
    #     super(empty, self).__init__()
    #     self.mean = aver_img(image).detach()
    #
    # def forward(self, img):
    #     # normalize img
    #     return img
    def __init__(self, mean, std):
        super(empty, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def gram_maxtrix(img):
    maxtrix = img.view(img.shape[0] * img.shape[1], img.shape[2] * img.shape[3])
    return torch.matmul(maxtrix, maxtrix.t())


class content(nn.Module):
    def __init__(self, target_img):
        super(content, self).__init__()
        self.target = target_img.detach()

    def forward(self, gen_img):
        self.loss = F.mse_loss(gen_img, self.target)
        return gen_img


class style(nn.Module):
    def __init__(self, target_img):
        super(style, self).__init__()
        self.target_gram = gram_maxtrix(target_img).detach()

    def forward(self, gen_img):
        gram_gen = gram_maxtrix(gen_img)
        self.loss = F.mse_loss(gram_gen, self.target_gram)
        return gen_img


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    utils.save_image(input_tensor, filename)


img_style = load_image(style_img_path)
img_content = load_image(content_img_path)
# img_content_clone = img_content.clone()
init_model = empty(cnn_normalization_mean,cnn_normalization_std)
# init_model = empty(img_content)
model_own = nn.Sequential(init_model)
mean_value = aver_img(img_content)

# 根据需要创建属于我们自己的模型，在vgg19的特定的卷子层后面加上损失
flag = 0
i = 0
style_loss_all = []
contert_loss_all = []

while (flag != 36):
    if isinstance(vgg19[flag], nn.ReLU):
        vgg19[flag] = nn.ReLU(inplace=False)  # 此处激活层的inpalce必须为False 否则无法优化
    if isinstance(vgg19[flag], nn.MaxPool2d):
        vgg19[flag] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    model_own.add_module('{}'.format(i + 1), vgg19[flag])
    if flag in STYLE_LAYER:
        i += 1
        target = model_own(img_style).detach()
        style_loss = style(target)
        model_own.add_module('{}'.format(i + 1), style_loss)
        style_loss_all.append(style_loss)

    if flag == CONTEN_LAYERS:
        i += 1
        target = model_own(img_style).detach()
        content_loss = content(target)
        model_own.add_module('{}'.format(i + 1), content_loss)
        contert_loss_all.append(content_loss)
    i += 1
    flag += 1

print(model_own)

optimizer = optim.Adam([img_content.requires_grad_()])  # 选择优化器

for m in range(epoch):
    img_content.data.clamp_(0, 1)
    optimizer.zero_grad()
    model_own(img_content)
    con_loss = 0
    sty_loss = 0
    i = 0

    for s in style_loss_all:
        sty_loss += s.loss * Weight[i]
        i += 1
    for sl in contert_loss_all:
        con_loss += sl.loss

    total_loss = con_loss * Weight_Content + sty_loss * Weight_Style

    total_loss.backward()
    optimizer.step()

    img_content.data.clamp_(0, 1)
    if m % 500 == 0:
        the_current_time()
        print("第", m, "轮:")
        print(total_loss)
        print('style_loss:{}'.format(sty_loss))
        print('content_loss:{}'.format(con_loss))
        save_image_tensor(img_content, os.path.join(save_path, 'output_%d.jpg' % m))
