# PuLID for FLUX
We are happy to release the **PuLID-FLUX-v0.9.0** model, which provides a tuning-free ID customization solution for FLUX.1-dev. 

If PuLID-FLUX is helpful, please help to ⭐ this repo or recommend it to your friends 😊

## Inference
### Local Gradio Demo
1. Please first follow the [dependencies-and-installation](../README.md#wrench-dependencies-and-installation) to set up the environment.
2. Download flux1-dev.safetensors and ae.safetensors from [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) to the models folder.
3. Run the gradio demo with `python app_flux.py`

### Online Demo
- huggingface demo: we are currently preparing it, will be available soon.

### ComfyUI
Please stay tuned for the community implementation

## Visual Results
![pulid_flux_results](https://github.com/user-attachments/assets/7eafb90a-fdd1-4ae7-bc41-8c428d568848)


## Useful Tips
There are two parameters that are crucial and need to be set carefully:

1. `timestep to start inserting ID`: This parameter controls the timing of ID insertion. If set to 0, the ID starts being inserted to the DIT from the first timestep. The earlier it is inserted, the higher the ID fidelity will be, but the editability may decrease. The later it is inserted, the lower the fidelity to the ID, but the editability will increase, and the disruption to the original model behavior will also be smaller. For generating realistic images, we suggest setting this to 4. If you found the ID similarity is not high enough, you could try lowering this parameter accordingly. For generating stylized images, we suggest setting it to 0-1.
![start_id](https://github.com/user-attachments/assets/3866ffab-542d-4e2f-9a0c-6877c9158d49)

2. `true CFG scale`: FLUX.1-dev is a guidance distill model. The original CFG process, which required twice the number of inference steps, is distilled into a guidance scale, thereby modulating the DIT through the guidance scale to simulate the true CFG process with half the inference steps. We will refer to this as fake CFG in the following doc. Our PuLID-FLUX model can be tested under the fake CFG settings, and the guidance scale can be set to a commonly used value, such as 4. However, the model also supports using the real CFG for inference. We compare the results of using true CFG with the fake CFG in photorealistic scenarios below.
![fake_cfg_vs_true_cfg_fidelity](https://github.com/user-attachments/assets/73b44dc8-37c7-48c8-8f55-73882731126d)
As shown in the above image, in terms of ID fidelity, using fake CFG is similar to true CFG in most cases, except that in a few cases, true CFG achieves higher ID similarity. In terms of image aesthetics and facial naturalness, fake CFG performs better. However, by carefully adjusting hyperparameters, the performance of true CFG may be further improved, we leave this to the community to explore. Therefore, we recommend using fake CFG for photorealistic scenes. If you are not satisfy about the ID fidelity, you can try switching to true CFG. Additionally, as shown below, we have found that using fake CFG in stylized scenes sometimes results in lower ID similarity and poorer style response, so if you encounter these two issues in stylized scenes, please consider switching to true CFG.
![fake_cfg_vs_true_cfg_style](https://github.com/user-attachments/assets/fb042639-64e6-4bb3-a3a4-5c138793318e)

   

## Some Technical Details
- We switch the ID encoder from an MLP structure to a Transformer structure. Interested users can refer to [source code](../pulid/encoders_flux.py)
- Inspired by [Flamingo](https://arxiv.org/abs/2204.14198), we insert additional cross-attention blocks every few DIT blocks to interact ID features with DIT image features
- We would like to clarify that the acceleration method (lile SDXL-Lightning) serves as an
optional acceleration trick, but it is not indispensable for training PuLID. We will update the arxiv paper with the relevant details in the near future. Please stay tuned.


## limitation
The model is currently in beta version, and we have observed that the ID fidelity may not be high for some male inputs, maybe the model requires more training. If the improved model is ready, we will release it here, so please stay tuned.

## contact
If you have any questions or suggestions about the model, please contact [Yanze Wu](https://tothebeginning.github.io/) or open an issue/discussion here.