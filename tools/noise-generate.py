from torchvision.transforms import ToPILImage, ToTensor
import torch
from PIL import Image

def random_noise(nc, width, height):
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    return img

def corrupt(image, amount=0.1):
    img = Image.open(image)
    img_tensor = ToTensor()(img)

    # 生成噪声
    noise = torch.randn_like(img_tensor)

    # 在原有的图像上添加噪声
    noise_img_tensor = img_tensor * (1 - amount) + noise * amount
    noise_img_tensor = torch.clamp(noise_img_tensor, 0, 1)

    # 转换为PIL图像
    noise_img = ToPILImage()(noise_img_tensor)

    return noise_img


if __name__ == '__main__':
    for i in range(6):
        img = corrupt('test.png', 0.2 * i)
        img.save('test_noise_{}.png'.format(i))