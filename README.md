# 人工智能导论自主学习
## StableDiffusion

Stable Diffusion(SD)模型是由Stability AI和LAION等公司共同开发的**生成式模型**，可以用于文生图、图生图、图像inpainting，ControlNet控制生成，图像超分等丰富的任务。

## Stable Diffusion原理
Stable Diffusion和GAN等生成式模型一样的是，SD模型同样学习拟合训练集分布，并能够生成与训练集分布相似的输出结果，但与它们不同的是SD模型的训练过程更加稳定，而且具备更强的泛化性能。

### 扩散模型
Stable Diffusion是一个扩散模型。扩散模型是通过神经网络学习从纯噪声数据逐渐对数据进行去噪的过程，从单个图像样来看，扩散过程就是不断往图像上添加噪声直到图像变成一个纯噪声。逆扩散过程就是从纯噪声生成一张图像的过程。它包含两个步骤：
1. 固定的前向过程$p$：在这一步逐渐将高斯噪声添加到图像中，直到得到一个纯噪声的图像；
2. 可学习的反向去噪过程$p_\theta$：在这一步从纯噪声图像中逐渐对其进行去噪，知道得到真实的图像

Diffusion模型用于生成与训练数据相似的数据。从根本上来说，Diffusion模型的工作原理，是通过连续添加高斯噪声来破坏训练数据，再反转这个过程，用于学习恢复数据。而训练目的便是让扩散模型每次预测出的噪声和每次实际加入的噪声做回归，让扩散模型能预测出每次实际加入的真实噪声。

## Stable Diffusion模型的网络架构
Stable Diffusion主要由VAE（变分自编码器， Variational Auto-Encoder），U-Net以及CLIP Text Encoder三个核心组件构成。

### VAE模型
VAE的Encoder结构能讲输入图像转化为低纬Latent特征，并作为U-Net的输入。VAE的Decoder结构能将低纬Latent特征重建还原成像素级图像。

VAE能够压缩数据，因此可以极大程度缩小计算复杂性，计算效率大大提升，同时也降低了硬件需求。正因如此，Stable Diffusion可以在较低配置的平台上运行。

VAE对图像的压缩可以看作是一个有损压缩，但由于自然图像并非完全随机，而是具有很高的规律性，因此这样的压缩并不会对图像的特征带来很大的损失。这也是模型能够使用VAE进行空间压缩的原因。在损失不大的前提下降低计算量，这便是VAE的作用

ldm/odel/autoencoder.py
``` python
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
```

其编码器与解码器定义在ldm/modules/diffusionmodules/model.py中的Encoder类与Decoder类，结构如下：

``` python
class Encoder:
# downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
```



``` python
class Decoder:
    # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
```



### U-Net模型
在Stable Diffusion中，U-Net模型是关键核心，其在模型中的主要作用是预测噪声残差，并结合调度算法对特征矩阵进行重构，从而将随机噪声转化成图片。



## Stable Diffusion模型的训练过程
StableDiffusion的训练过程可以看作是一个加噪声与去噪声的过程，在这个针对噪声的对抗过程中学习生成图片的能力

模型的整体训练逻辑为：
1. 从数据集中随机选择训练样本
2. 从K歌噪声量级中随机抽样一个timestep
3. 产生随机噪声
4. 计算当前产生的噪声数据
5. 将噪声输入**U-Net**预测噪声
6. 计算产生的噪声和预测的噪声的L2损失
7. 计算梯度并进行参数更新

