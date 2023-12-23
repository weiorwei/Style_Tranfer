import os
import imageio
import torch
import torchvision
from skimage.transform import resize
from torch import nn
import numpy as np
import time
import torch.optim as optim
import torchvision.transforms as transforms

# 加载vgg19模型


weight = torchvision.models.VGG19_Weights.IMAGENET1K_V1  # vgg19 模型参数
vgg19 = torchvision.models.vgg19(pretrained=True).to('cuda').eval()  # 加载vgg19 模型
# vgg19.eval()
print(vgg19)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         weight = torchvision.models.VGG19_Weights.IMAGENET1K_V1  # vgg19 模型参数
#         self.model = torchvision.models.vgg19(weights=weight)  # 加载vgg19 模型
#         self.model.features[4] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
#
# own = Net()


# 以下为可以调的一些初始化参数
nosie_ratio = 0.5
style_img_path = 'style_pic/sleep_flower.jpg'
content_img_path = 'origin_pic/proof.jpg'
weight_style = 100
weight_content = 5
IMAGE_HIGHT = 30
IMAGE_WIDTH = 40
IMAGE_C = 3
epoch = 2000
save_path = 'neural_style_transfer_own/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


# ------------------------------


# 打印时间
def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))


# 获得模型的每一层输出结果，输出结果为一个列表  例： pre[0]为第一层的输出
def pick_conv(img, model, max_layer):
    pre = []
    pre.append(img)
    for i in range(max_layer):
        pre.append(model.features[i](pre[i]).detach())
    return pre


def content_loss(origin_img, genera_img, compare_layer):
    pre_orig = pick_conv(origin_img.cuda(), vgg19, compare_layer)  # 将原始图片输入模型获得每一层的输出
    pre_genera_img = pick_conv(genera_img.cuda(), vgg19, compare_layer)  # 将生成的图片图片输入模型获得每一层的输出
    x = pre_orig[
        compare_layer]  # 取模型的第compare_layer层 例如第十二层(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 的输出作为内容比较的
    y = pre_genera_img[compare_layer]
    area = x.shape[2] * x.shape[3]  # 获得图像的面积
    height = x.shape[1]  # 获得图像的深度
    m = (x - y)
    content_loss = (1 / (4 * area * height)) * torch.sum(m ** 2)
    return torch.as_tensor(content_loss)


Weight_LAYERS = [(0.5, 1), (1.0, 5), (1.5, 10), (3.0, 19), (4.0, 34)]  # 选择计算gram矩阵的卷积层，以及对应的权重


def style_loss(style_img, genera_img):
    _style_loss = 0.0

    def gram_matrix(pre, H, area):
        maxtrix = torch.reshape(pre, (H, area))  # 将三维矩阵转换成2维矩阵
        return torch.matmul(maxtrix.transpose(0,1), maxtrix)  # 计算gram矩阵，两个矩阵转置相乘

    pre_style = pick_conv(style_img.cuda(), vgg19, 36)  # 将原始图片输入模型获得每一层的输出
    pre_genera_img = pick_conv(genera_img.cuda(), vgg19, 36)  # 将生成的图片图片输入模型获得每一层的输出
    for w, layer in Weight_LAYERS:
        H1 = pre_style[layer].shape[1]
        area1 = pre_style[layer].shape[2] * pre_style[layer].shape[3]
        gram_style = gram_matrix(pre_style[layer], H1, area1)
        H2 = pre_genera_img[layer].shape[1]
        area2 = pre_genera_img[layer].shape[2] * pre_genera_img[layer].shape[3]
        gram_genera_img = gram_matrix(pre_genera_img[layer], H2, area2)
        a = gram_style - gram_genera_img
        b = a ** 2
        # b = b.numpy()
        _style_loss += w * (1 / (4 * area1 ** 2 * H1 ** 2) * torch.sum(b))  # np.sum() 必须是numpy的数据
    return torch.as_tensor(_style_loss)


# 产生噪声图片输入图片的格式必须为(batch_size(一般为1),通道数，宽，高)
def generate_img(img, noise_rate):
    # noise = np.random.uniform(-20, 20, (1, img.shape[1], img.shape[2], img.shape[3])).astype('float32')
    # img_gen = torch.from_numpy(noise) * noise_rate + img * (1 - noise_rate)
    img_gen = img
    return img_gen


# 加载图片
def load_img(path):
    def aver_img():
        # 计算图片的平均值
        image = imageio.imread(path)
        image = resize(image, (IMAGE_HIGHT, IMAGE_WIDTH), preserve_range=True)
        MEAN_R = np.mean(image[:, :, 0])
        MEAN_G = np.mean(image[:, :, 1])
        MEAN_B = np.mean(image[:, :, 2])
        MEAN_VALUES = torch.ones((image.shape[0], image.shape[1], image.shape[2]))
        MEAN_VALUES[:, :, 0] = MEAN_VALUES[:, :, 0] * MEAN_R
        MEAN_VALUES[:, :, 1] = MEAN_VALUES[:, :, 1] * MEAN_G
        MEAN_VALUES[:, :, 2] = MEAN_VALUES[:, :, 2] * MEAN_B
        return image, MEAN_VALUES

    img, mean_img = aver_img()
    img = torch.from_numpy(img)  # 将图片从numpy.ndarray转换成Tensor数据类型

    img = img - mean_img
    img = np.transpose(img, (2, 0, 1))
    img = img.unsqueeze(0)
    return img, mean_img


def save_img(path, img, mean_img):
    img = img + mean_img
    img = img[0]
    img_save = img.detach().numpy()
    img_save = np.clip(img_save, 0, 255).astype('uint8')
    imageio.imsave(path, img_save)


class gen_img(nn.Module):
    def __init__(self):
        super(gen_img, self).__init__()
        # self.model=nn.Sequential{
        #     nn.
        # }
        # self.img = generate_img(img, nosie_ratio)
        self.noise = torch.randn((3, IMAGE_HIGHT, IMAGE_WIDTH))
        # self.img_gen = torch.from_numpy(self.noise) * nosie_ratio + img * (1 - nosie_ratio)

    def forward(self, img):
        x = img + self.noise
        return x


class content(nn.Module):
    def __init__(self):
        super(content, self).__init__()

    def forward(self, ori_img, gen_img, layer):
        x = content_loss(ori_img, gen_img, layer)
        return x


class style(nn.Module):
    def __init__(self):
        super(style, self).__init__()

    def forward(self, ori_img, gen_img):
        z = style_loss(ori_img, gen_img)
        return z


the_current_time()

content_img, content_img_mean = load_img(content_img_path)
style_img, style_img_mean = load_img(style_img_path)
input = generate_img(content_img, nosie_ratio)
# input_class = gen_img(content_img)
# input = input_class(content_img)
input_c = input.clone()
optimizer = optim.Adam([input_c.requires_grad_(True)], lr=0.00001)  # 选择优化器
# 实例化两个损失函数
loss_content_class = content()
loss_style_class = style()

for i in range(epoch):
    loss_style = loss_style_class(style_img.to(torch.float32), input.to(torch.float32)).to('cuda')  # 风格损失
    loss_content = loss_content_class(content_img.to(torch.float32), input.to(torch.float32), 19).to('cuda')  # 内容损失
    total_loss = loss_style * weight_style + loss_content * weight_content
    total_loss.requires_grad_(True)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print("第", i, "轮:")
        print(total_loss)
        if content_img_mean.shape[0] != 3:
            content_img_mean = np.transpose(content_img_mean, (2, 0, 1))
        input_c = input.squeeze(0)
        save_img(os.path.join(save_path, 'output_%d.jpg' % i), input_c, content_img_mean)
