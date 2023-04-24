![](asset/bg.jpg)

# MuG Diffusion

MuG Diffusion 是一款基于 [Stable Diffusion](https://github.com/CompVis/latent-diffusion/) （最强大的AIGC模型之一） 的节奏游戏谱面制作AI，并进行了大量修改以整合音频波。给定一个音频文件，MuG Diffusion 能够生成高质量的多样化谱面，这些谱面与音乐保持一致，并且高度可控。目前，它仅支持4K垂直下落式节奏游戏（VSRG），有以下控制选项：

- 难度：同时支持 [osu! star rating system](https://osu.ppy.sh/wiki/en/Beatmap/Star_rating) 和 [Etterna MSD system](https://etternaonline.com/) 。
- 风格：ranked 谱面（[osu!](https://osu.ppy.sh/)） / stable 谱面 （[Malody](https://m.mugzone.net/)） ，与其它谱面风格。 
- 长音符：长音符的物量与总物量的比率。
- 模式：支持在 Etterna MSD 系统的所有模式，包括 chordjack，stamina，stream，jumpstream，handstream 和 technical。

MuG Diffusion 的目标是未来支持其它节奏游戏（osu!standard，下落式5-8K，maimai等），并希望为所有谱师和玩家提供一个有益的AIGC工具。

![](asset/screenshot1.png)
![](asset/screenshot2.png)

## 安装与运行

### 压缩整合包

我打包了一份包含 Windows 平台中所有的依赖项和模型权重，双击即可直接运行的整合包，可以在 [这里](https://mug-diffusion-1305818561.cos.ap-nanjing.myqcloud.com/MugDiffusion.zip) 获取。
解压整合包并双击“Mug Diffusion.exe”，将会自动打开浏览器界面进行控制。在我的电脑（NVidia RTX 3050Ti，4GB显存）上，为一条3分钟长的音频生成四张谱面大约需要30秒。


### 运行源代码

如果您使用其它平台或希望从源代码运行，以下为说明。

- 安装 Python 。

- 安装 pytorch-cuda ： https://pytorch.org/get-started/locally/

- 安装其它依赖包：

```commandline
pip install -r requirements.txt
```

- 下载整合包，并复制文件 `models/ckpt/model.ckpt` 和 `models/ckpt/model.yaml` 到 `{REPOSITORY_ROOT}/models/ckpt/*` 目录下。

- 运行网页控制台：

```commandline
python webui.py
```

## 模型结构与方法

未完待续

## 致谢与相关人员

感谢社区中的每一位谱师。是你们赋予了 MuG Diffusion 智慧。此外，我还要感谢 [Malody](https://m.mugzone.net/) 开发团队（以及许多其它由于篇幅限制而无法列出的支持者 TAT）的财力支持。

感谢 [raber](https://github.com/zengrber) 对网页控制台的开发， [RiceSS](https://osu.ppy.sh/users/8271436) 的图标设计，与许多测试人员的支持。

特别感谢： 
- [kangalio](https://github.com/kangalio/)：来源于 [MinaCalc-standalone](https://github.com/kangalio/minacalc-standalone)，MSD控制系统的一个组件。
- [Evening](ttps://github.com/Eve-ning/)：来源于 [reamberPy](https://github.com/Eve-ning/reamberPy)，提供了谱面生成非常直观的可视化预览。

通过 Mug Diffusion 生成的谱面完全开源，明确遵循 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 通用公共领域专用协议。
