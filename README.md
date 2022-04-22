[comment]: <> (## Training and testing codes for USRNet, DnCNN, FFDNet, SRMD, DPSR, MSRResNet, ESRGAN, BSRGAN, SwinIR)

[comment]: <> ([![download]&#40;https://img.shields.io/github/downloads/cszn/KAIR/total.svg&#41;]&#40;https://github.com/cszn/KAIR/releases&#41; ![visitors]&#40;https://visitor-badge.glitch.me/badge?page_id=cszn/KAIR&#41; )

[comment]: <> ([Kai Zhang]&#40;https://cszn.github.io/&#41;)

[comment]: <> (*[Computer Vision Lab]&#40;https://vision.ee.ethz.ch/the-institute.html&#41;, ETH Zurich, Switzerland*)

[comment]: <> (_______)

[comment]: <> (- **_News &#40;2021-12-23&#41;_**: Our techniques are adopted in [https://www.amemori.ai/]&#40;https://www.amemori.ai/&#41;.)

[comment]: <> (- **_News &#40;2021-12-23&#41;_**: Our new work for practical image denoising.)

[comment]: <> (- <img src="figs/palace.png" height="320px"/> <img src="figs/palace_HSCU.png" height="320px"/> )

[comment]: <> (- [<img src="https://github.com/cszn/KAIR/raw/master/figs/denoising_02.png" height="256px"/>]&#40;https://imgsli.com/ODczMTc&#41; )

[comment]: <> ([<img src="https://github.com/cszn/KAIR/raw/master/figs/denoising_01.png" height="256px"/>]&#40;https://imgsli.com/ODczMTY&#41; )

[comment]: <> (- **_News &#40;2021-09-09&#41;_**: Add [main_download_pretrained_models.py]&#40;https://github.com/cszn/KAIR/blob/master/main_download_pretrained_models.py&#41; to download pre-trained models.)

[comment]: <> (- **_News &#40;2021-09-08&#41;_**: Add [matlab code]&#40;https://github.com/cszn/KAIR/tree/master/matlab&#41; to zoom local part of an image for the purpose of comparison between different results.)

[comment]: <> (- **_News &#40;2021-09-07&#41;_**: We upload [the training code]&#40;https://github.com/cszn/KAIR/blob/master/docs/README_SwinIR.md&#41; of [SwinIR ![GitHub Stars]&#40;https://img.shields.io/github/stars/JingyunLiang/SwinIR?style=social&#41;]&#40;https://github.com/JingyunLiang/SwinIR&#41; and provide an [interactive online Colob demo for real-world image SR]&#40;https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb&#41;. Try to super-resolve your own images on Colab! <a href="https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>)

[comment]: <> (|Real-World Image &#40;x4&#41;|[BSRGAN, ICCV2021]&#40;https://github.com/cszn/BSRGAN&#41;|[Real-ESRGAN]&#40;https://github.com/xinntao/Real-ESRGAN&#41;|SwinIR &#40;ours&#41;|)

[comment]: <> (|      :---      |     :---:        |        :-----:         |        :-----:         | )

[comment]: <> (|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_LR.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_BSRGAN.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_realESRGAN.jpg">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/ETH_SwinIR.png">)

[comment]: <> (|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_LR.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_BSRGAN.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_realESRGAN.png">|<img width="200" src="https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/figs/OST_009_crop_SwinIR.png">|)

[comment]: <> (- **_News &#40;2021-08-31&#41;_**: We upload the [training code of BSRGAN]&#40;https://github.com/cszn/BSRGAN#training&#41;.)

[comment]: <> (- **_News &#40;2021-08-24&#41;_**: We upload the BSRGAN degradation model.)

[comment]: <> (- **_News &#40;2021-08-22&#41;_**: Support multi-feature-layer VGG perceptual loss and UNet discriminator. )

[comment]: <> (- **_News &#40;2021-08-18&#41;_**: We upload the extended BSRGAN degradation model. It is slightly different from our published version. )

[comment]: <> (- **_News &#40;2021-06-03&#41;_**: Add testing codes of [GPEN &#40;CVPR21&#41;]&#40;https://github.com/yangxy/GPEN&#41; for face image enhancement: [main_test_face_enhancement.py]&#40;https://github.com/cszn/KAIR/blob/master/main_test_face_enhancement.py&#41;)

[comment]: <> (<img src="figs/face_04_comparison.png" width="730px"/> )

[comment]: <> (<img src="figs/face_13_comparison.png" width="730px"/> )

[comment]: <> (<img src="figs/face_08_comparison.png" width="730px"/> )

[comment]: <> (<img src="figs/face_01_comparison.png" width="730px"/> )

[comment]: <> (<img src="figs/face_12_comparison.png" width="730px"/> )

[comment]: <> (<img src="figs/face_10_comparison.png" width="730px"/> )


[comment]: <> (- **_News &#40;2021-05-13&#41;_**: Add [PatchGAN discriminator]&#40;https://github.com/cszn/KAIR/blob/master/models/network_discriminator.py&#41;.)

[comment]: <> (- **_News &#40;2021-05-12&#41;_**: Support distributed training, see also [https://github.com/xinntao/BasicSR/blob/master/docs/TrainTest.md]&#40;https://github.com/xinntao/BasicSR/blob/master/docs/TrainTest.md&#41;.)

[comment]: <> (- **_News &#40;2021-01&#41;_**: [BSRGAN]&#40;https://github.com/cszn/BSRGAN&#41; for blind real image super-resolution will be added.)

[comment]: <> (- **_Pull requests are welcome!_**)

[comment]: <> (- **Correction &#40;2020-10&#41;**: If you use multiple GPUs for GAN training, remove or comment [Line 105]&#40;https://github.com/cszn/KAIR/blob/e52a6944c6a40ba81b88430ffe38fd6517e0449e/models/model_gan.py#L105&#41; to enable `DataParallel` for fast training)

[comment]: <> (- **News &#40;2020-10&#41;**: Add [utils_receptivefield.py]&#40;https://github.com/cszn/KAIR/blob/master/utils/utils_receptivefield.py&#41; to calculate receptive field.)

[comment]: <> (- **News &#40;2020-8&#41;**: A `deep plug-and-play image restoration toolbox` is released at [cszn/DPIR]&#40;https://github.com/cszn/DPIR&#41;.)

[comment]: <> (- **Tips &#40;2020-8&#41;**: Use [this]&#40;https://github.com/cszn/KAIR/blob/9fd17abff001ab82a22070f7e442bb5246d2d844/main_challenge_sr.py#L147&#41; to avoid `out of memory` issue.)

[comment]: <> (- **News &#40;2020-7&#41;**: Add [main_challenge_sr.py]&#40;https://github.com/cszn/KAIR/blob/23b0d0f717980e48fad02513ba14045d57264fe1/main_challenge_sr.py#L90&#41; to get `FLOPs`, `#Params`, `Runtime`, `#Activations`, `#Conv`, and `Max Memory Allocated`.)

[comment]: <> (```python)

[comment]: <> (from utils.utils_modelsummary import get_model_activation, get_model_flops)

[comment]: <> (input_dim = &#40;3, 256, 256&#41;  # set the input dimension)

[comment]: <> (activations, num_conv2d = get_model_activation&#40;model, input_dim&#41;)

[comment]: <> (logger.info&#40;'{:>16s} : {:<.4f} [M]'.format&#40;'#Activations', activations/10**6&#41;&#41;)

[comment]: <> (logger.info&#40;'{:>16s} : {:<d}'.format&#40;'#Conv2d', num_conv2d&#41;&#41;)

[comment]: <> (flops = get_model_flops&#40;model, input_dim, False&#41;)

[comment]: <> (logger.info&#40;'{:>16s} : {:<.4f} [G]'.format&#40;'FLOPs', flops/10**9&#41;&#41;)

[comment]: <> (num_parameters = sum&#40;map&#40;lambda x: x.numel&#40;&#41;, model.parameters&#40;&#41;&#41;&#41;)

[comment]: <> (logger.info&#40;'{:>16s} : {:<.4f} [M]'.format&#40;'#Params', num_parameters/10**6&#41;&#41;)

[comment]: <> (```)

[comment]: <> (- **News &#40;2020-6&#41;**: Add [USRNet &#40;CVPR 2020&#41;]&#40;https://github.com/cszn/USRNet&#41; for training and testing.)

[comment]: <> (  - [Network Architecture]&#40;https://github.com/cszn/KAIR/blob/3357aa0e54b81b1e26ceb1cee990f39add235e17/models/network_usrnet.py#L309&#41;)

[comment]: <> (  - [Dataset]&#40;https://github.com/cszn/KAIR/blob/6c852636d3715bb281637863822a42c72739122a/data/dataset_usrnet.py#L16&#41;)


[comment]: <> (Clone repo)

[comment]: <> (----------)

[comment]: <> (```)

[comment]: <> (git clone https://github.com/cszn/KAIR.git)

[comment]: <> (```)

[comment]: <> (```)

[comment]: <> (pip install -r requirement.txt)

[comment]: <> (```)



[comment]: <> (Training)

[comment]: <> (----------)

[comment]: <> (You should modify the json file from [options]&#40;https://github.com/cszn/KAIR/tree/master/options&#41; first, for example,)

[comment]: <> (setting ["gpu_ids": [0,1,2,3]]&#40;https://github.com/cszn/KAIR/blob/ff80d265f64de67dfb3ffa9beff8949773c81a3d/options/train_msrresnet_psnr.json#L4&#41; if 4 GPUs are used,)

[comment]: <> (setting ["dataroot_H": "trainsets/trainH"]&#40;https://github.com/cszn/KAIR/blob/ff80d265f64de67dfb3ffa9beff8949773c81a3d/options/train_msrresnet_psnr.json#L24&#41; if path of the high quality dataset is `trainsets/trainH`.)

[comment]: <> (- Training with `DataParallel` - PSNR)


[comment]: <> (```python)

[comment]: <> (python main_train_psnr.py --opt options/train_msrresnet_psnr.json)

[comment]: <> (```)

[comment]: <> (- Training with `DataParallel` - GAN)

[comment]: <> (```python)

[comment]: <> (python main_train_gan.py --opt options/train_msrresnet_gan.json)

[comment]: <> (```)

[comment]: <> (- Training with `DistributedDataParallel` - PSNR - 4 GPUs)

[comment]: <> (```python)

[comment]: <> (python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/train_msrresnet_psnr.json  --dist True)

[comment]: <> (```)

[comment]: <> (- Training with `DistributedDataParallel` - PSNR - 8 GPUs)

[comment]: <> (```python)

[comment]: <> (python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/train_msrresnet_psnr.json  --dist True)

[comment]: <> (```)

[comment]: <> (- Training with `DistributedDataParallel` - GAN - 4 GPUs)

[comment]: <> (```python)

[comment]: <> (python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_gan.py --opt options/train_msrresnet_gan.json  --dist True)

[comment]: <> (```)

[comment]: <> (- Training with `DistributedDataParallel` - GAN - 8 GPUs)

[comment]: <> (```python)

[comment]: <> (python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_gan.py --opt options/train_msrresnet_gan.json  --dist True)

[comment]: <> (```)

[comment]: <> (- Kill distributed training processes of `main_train_gan.py`)

[comment]: <> (```python)

[comment]: <> (kill $&#40;ps aux | grep main_train_gan.py | grep -v grep | awk '{print $2}'&#41;)

[comment]: <> (```)

[comment]: <> (----------)

[comment]: <> (| Method | Original Link |)

[comment]: <> (|---|---|)

[comment]: <> (| DnCNN |[https://github.com/cszn/DnCNN]&#40;https://github.com/cszn/DnCNN&#41;|)

[comment]: <> (| FDnCNN |[https://github.com/cszn/DnCNN]&#40;https://github.com/cszn/DnCNN&#41;|)

[comment]: <> (| FFDNet | [https://github.com/cszn/FFDNet]&#40;https://github.com/cszn/FFDNet&#41;|)

[comment]: <> (| SRMD | [https://github.com/cszn/SRMD]&#40;https://github.com/cszn/SRMD&#41;|)

[comment]: <> (| DPSR-SRResNet | [https://github.com/cszn/DPSR]&#40;https://github.com/cszn/DPSR&#41;|)

[comment]: <> (| SRResNet | [https://github.com/xinntao/BasicSR]&#40;https://github.com/xinntao/BasicSR&#41;|)

[comment]: <> (| ESRGAN | [https://github.com/xinntao/ESRGAN]&#40;https://github.com/xinntao/ESRGAN&#41;|)

[comment]: <> (| RRDB | [https://github.com/xinntao/ESRGAN]&#40;https://github.com/xinntao/ESRGAN&#41;|)

[comment]: <> (| IMDB | [https://github.com/Zheng222/IMDN]&#40;https://github.com/Zheng222/IMDN&#41;|)

[comment]: <> (| USRNet | [https://github.com/cszn/USRNet]&#40;https://github.com/cszn/USRNet&#41;|)

[comment]: <> (| DRUNet | [https://github.com/cszn/DPIR]&#40;https://github.com/cszn/DPIR&#41;|)

[comment]: <> (| DPIR | [https://github.com/cszn/DPIR]&#40;https://github.com/cszn/DPIR&#41;|)

[comment]: <> (| BSRGAN | [https://github.com/cszn/BSRGAN]&#40;https://github.com/cszn/BSRGAN&#41;|)

[comment]: <> (| SwinIR | [https://github.com/JingyunLiang/SwinIR]&#40;https://github.com/JingyunLiang/SwinIR&#41;|)

[comment]: <> (Network architectures)

[comment]: <> (----------)

[comment]: <> (* [USRNet]&#40;https://github.com/cszn/USRNet&#41;)

[comment]: <> (  <img src="https://github.com/cszn/USRNet/blob/master/figs/architecture.png" width="600px"/> )

[comment]: <> (* DnCNN)

[comment]: <> (  <img src="https://github.com/cszn/DnCNN/blob/master/figs/dncnn.png" width="600px"/> )
 
[comment]: <> (* IRCNN denoiser)

[comment]: <> ( <img src="https://github.com/lipengFu/IRCNN/raw/master/Image/image_2.png" width="680px"/> )

[comment]: <> (* FFDNet)

[comment]: <> (  <img src="https://github.com/cszn/FFDNet/blob/master/figs/ffdnet.png" width="600px"/> )

[comment]: <> (* SRMD)

[comment]: <> (  <img src="https://github.com/cszn/SRMD/blob/master/figs/architecture.png" width="605px"/> )

[comment]: <> (* SRResNet, SRGAN, RRDB, ESRGAN)

[comment]: <> (  <img src="https://github.com/xinntao/ESRGAN/blob/master/figures/architecture.jpg" width="595px"/> )
  
[comment]: <> (* IMDN)

[comment]: <> (  <img src="figs/imdn.png" width="460px"/>  ----- <img src="figs/imdn_block.png" width="100px"/> )



[comment]: <> (Testing)

[comment]: <> (----------)

[comment]: <> (|Method | [model_zoo]&#40;model_zoo&#41;|)

[comment]: <> (|---|---|)

[comment]: <> (| [main_test_dncnn.py]&#40;main_test_dncnn.py&#41; |```dncnn_15.pth, dncnn_25.pth, dncnn_50.pth, dncnn_gray_blind.pth, dncnn_color_blind.pth, dncnn3.pth```|)

[comment]: <> (| [main_test_ircnn_denoiser.py]&#40;main_test_ircnn_denoiser.py&#41; | ```ircnn_gray.pth, ircnn_color.pth```| )

[comment]: <> (| [main_test_fdncnn.py]&#40;main_test_fdncnn.py&#41; | ```fdncnn_gray.pth, fdncnn_color.pth, fdncnn_gray_clip.pth, fdncnn_color_clip.pth```|)

[comment]: <> (| [main_test_ffdnet.py]&#40;main_test_ffdnet.py&#41; | ```ffdnet_gray.pth, ffdnet_color.pth, ffdnet_gray_clip.pth, ffdnet_color_clip.pth```|)

[comment]: <> (| [main_test_srmd.py]&#40;main_test_srmd.py&#41; | ```srmdnf_x2.pth, srmdnf_x3.pth, srmdnf_x4.pth, srmd_x2.pth, srmd_x3.pth, srmd_x4.pth```| )

[comment]: <> (|  | **The above models are converted from MatConvNet.** |)

[comment]: <> (| [main_test_dpsr.py]&#40;main_test_dpsr.py&#41; | ```dpsr_x2.pth, dpsr_x3.pth, dpsr_x4.pth, dpsr_x4_gan.pth```|)

[comment]: <> (| [main_test_msrresnet.py]&#40;main_test_msrresnet.py&#41; | ```msrresnet_x4_psnr.pth, msrresnet_x4_gan.pth```|)

[comment]: <> (| [main_test_rrdb.py]&#40;main_test_rrdb.py&#41; | ```rrdb_x4_psnr.pth, rrdb_x4_esrgan.pth```|)

[comment]: <> (| [main_test_imdn.py]&#40;main_test_imdn.py&#41; | ```imdn_x4.pth```|)

[comment]: <> ([model_zoo]&#40;model_zoo&#41;)

[comment]: <> (--------)

[comment]: <> (- download link [https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D]&#40;https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D&#41;)

[comment]: <> ([trainsets]&#40;trainsets&#41;)

[comment]: <> (----------)

[comment]: <> (- [https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md]&#40;https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md&#41;)

[comment]: <> (- [train400]&#40;https://github.com/cszn/DnCNN/tree/master/TrainingCodes/DnCNN_TrainingCodes_v1.0/data&#41;)

[comment]: <> (- [DIV2K]&#40;https://data.vision.ee.ethz.ch/cvl/DIV2K/&#41;)

[comment]: <> (- [Flickr2K]&#40;https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar&#41;)

[comment]: <> (- optional: use [split_imageset&#40;original_dataroot, taget_dataroot, n_channels=3, p_size=512, p_overlap=96, p_max=800&#41;]&#40;https://github.com/cszn/KAIR/blob/3ee0bf3e07b90ec0b7302d97ee2adb780617e637/utils/utils_image.py#L123&#41; to get ```trainsets/trainH``` with small images for fast data loading)

[comment]: <> ([testsets]&#40;testsets&#41;)

[comment]: <> (-----------)

[comment]: <> (- [https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md]&#40;https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md&#41;)

[comment]: <> (- [set12]&#40;https://github.com/cszn/FFDNet/tree/master/testsets&#41;)

[comment]: <> (- [bsd68]&#40;https://github.com/cszn/FFDNet/tree/master/testsets&#41;)

[comment]: <> (- [cbsd68]&#40;https://github.com/cszn/FFDNet/tree/master/testsets&#41;)

[comment]: <> (- [kodak24]&#40;https://github.com/cszn/FFDNet/tree/master/testsets&#41;)

[comment]: <> (- [srbsd68]&#40;https://github.com/cszn/DPSR/tree/master/testsets/BSD68/GT&#41;)

[comment]: <> (- set5)

[comment]: <> (- set14)

[comment]: <> (- cbsd100)

[comment]: <> (- urban100)

[comment]: <> (- manga109)


[comment]: <> (References)

[comment]: <> (----------)

[comment]: <> (```BibTex)

[comment]: <> (@inproceedings{liang2021swinir,)

[comment]: <> (title={SwinIR: Image Restoration Using Swin Transformer},)

[comment]: <> (author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},)

[comment]: <> (booktitle={IEEE International Conference on Computer Vision Workshops},)

[comment]: <> (year={2021})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2021designing,)

[comment]: <> (title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},)

[comment]: <> (author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},)

[comment]: <> (booktitle={IEEE International Conference on Computer Vision},)

[comment]: <> (year={2021})

[comment]: <> (})

[comment]: <> (@article{zhang2021plug, % DPIR & DRUNet & IRCNN)

[comment]: <> (  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},)

[comment]: <> (  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},)

[comment]: <> (  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},)

[comment]: <> (  year={2021})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2020aim, % efficientSR_challenge)

[comment]: <> (  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},)

[comment]: <> (  author={Kai Zhang and Martin Danelljan and Yawei Li and Radu Timofte and others},)

[comment]: <> (  booktitle={European Conference on Computer Vision Workshops},)

[comment]: <> (  year={2020})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2020deep, % USRNet)

[comment]: <> (  title={Deep unfolding network for image super-resolution},)

[comment]: <> (  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},)

[comment]: <> (  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},)

[comment]: <> (  pages={3217--3226},)

[comment]: <> (  year={2020})

[comment]: <> (})

[comment]: <> (@article{zhang2017beyond, % DnCNN)

[comment]: <> (  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},)

[comment]: <> (  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},)

[comment]: <> (  journal={IEEE Transactions on Image Processing},)

[comment]: <> (  volume={26},)

[comment]: <> (  number={7},)

[comment]: <> (  pages={3142--3155},)

[comment]: <> (  year={2017})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2017learning, % IRCNN)

[comment]: <> (title={Learning deep CNN denoiser prior for image restoration},)

[comment]: <> (author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},)

[comment]: <> (booktitle={IEEE conference on computer vision and pattern recognition},)

[comment]: <> (pages={3929--3938},)

[comment]: <> (year={2017})

[comment]: <> (})

[comment]: <> (@article{zhang2018ffdnet, % FFDNet, FDnCNN)

[comment]: <> (  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},)

[comment]: <> (  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},)

[comment]: <> (  journal={IEEE Transactions on Image Processing},)

[comment]: <> (  volume={27},)

[comment]: <> (  number={9},)

[comment]: <> (  pages={4608--4622},)

[comment]: <> (  year={2018})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2018learning, % SRMD)

[comment]: <> (  title={Learning a single convolutional super-resolution network for multiple degradations},)

[comment]: <> (  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},)

[comment]: <> (  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},)

[comment]: <> (  pages={3262--3271},)

[comment]: <> (  year={2018})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2019deep, % DPSR)

[comment]: <> (  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},)

[comment]: <> (  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},)

[comment]: <> (  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},)

[comment]: <> (  pages={1671--1681},)

[comment]: <> (  year={2019})

[comment]: <> (})

[comment]: <> (@InProceedings{wang2018esrgan, % ESRGAN, MSRResNet)

[comment]: <> (    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},)

[comment]: <> (    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},)

[comment]: <> (    booktitle = {The European Conference on Computer Vision Workshops &#40;ECCVW&#41;},)

[comment]: <> (    month = {September},)

[comment]: <> (    year = {2018})

[comment]: <> (})

[comment]: <> (@inproceedings{hui2019lightweight, % IMDN)

[comment]: <> (  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},)

[comment]: <> (  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},)

[comment]: <> (  booktitle={Proceedings of the 27th ACM International Conference on Multimedia &#40;ACM MM&#41;},)

[comment]: <> (  pages={2024--2032},)

[comment]: <> (  year={2019})

[comment]: <> (})

[comment]: <> (@inproceedings{zhang2019aim, % IMDN)

[comment]: <> (  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},)

[comment]: <> (  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},)

[comment]: <> (  booktitle={IEEE International Conference on Computer Vision Workshops},)

[comment]: <> (  year={2019})

[comment]: <> (})

[comment]: <> (@inproceedings{yang2021gan,)

[comment]: <> (    title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},)

[comment]: <> (    author={Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang},)

[comment]: <> (    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},)

[comment]: <> (    year={2021})

[comment]: <> (})

[comment]: <> (```)
