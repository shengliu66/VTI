<div align="center">   
  
# Reducing Hallucinations in Vision-Language Models via Latent Space Steering
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2007.00151-green)](https://arxiv.org/abs/2410.15778)

</div>

This repository is the official implementation of [Reducing Hallucinations in Vision-Language Models via Latent Space Steering
](https://arxiv.org/abs/2410.15778).

## 🎯 Overview

Hallucination poses a challenge to the deployment of large vision-language models (LVLMs) in applications. Unlike in large language models (LLMs), hallucination in LVLMs often arises from misalignments between visual inputs and textual outputs. This paper investigates the underlying mechanisms of hallucination, focusing on the unique structure of LVLMs that distinguishes them from large language models (LLMs). We identify that hallucinations often arise from the sensitivity of text decoders to vision inputs, a natural phenomenon when image encoders and text decoders are pre-trained separately. Inspired by this, we introduce Visual and Textual Intervention (VTI), a novel technique designed to reduce hallucinations by steering latent space representations during inference to enhance the stability of vision features. As a task-agnostic test-time intervention, VTI can be easily applied to any problem without additional cost. Extensive experiments demonstrate that it can effectively reduce hallucinations and outperform baseline methods across multiple metrics, highlighting the critical role of vision feature stability in LVLMs.

<p float="left" align="center">
<img src="images/vti_overview.png" width="800" /> 
<figcaption align="center">
Overview of the proposed algorithm visual and textual test-time intervention (VTI). Given an example set 
{(vᵢ, xᵢ, x̅ᵢ)} where vᵢ is the vision input and (xᵢ, x̅ᵢ) is paired captions with and without hallucination, VTI first runs the model on each query (vᵢ, xᵢ, x̅ᵢ) and records all hidden states. It then computes the shifting vectors dₗ,ₜᵛⁱˢⁱᵒⁿ and dₗ,ₜᵗᵉˣᵗ for all layer l and token t according to the method section in the paper. During inference, the vectors are subsequently added to every layer of the vision encoder and text decoder, respectively, when processing a new query. Notice that the vectors are task- and dataset-agnostic, i.e., they are pre-computed using a few samples from one specific task and dataset, and fixed unchanged throughout the entire experiments in our paper.
</figcaption>
</p>

## 🕹️ Usage

```
conda create -yn vti python=3.9
conda activate vti
cd VTI
pip install -r requirements.txt
```

### Data
The following evaluation requires for MSCOCO 2014 dataset (for computing the VTI directions as well as evaluation). Please download [here](https://cocodataset.org/#home) and extract it in your data path.

### How to Use VTI in LVLMs
There are two core functions of VTI, computing the VTI directions and adding the directions to the LVLM.
1. Compute the VTI visual and textual directions for a LVLM model
```
input_images, input_ids = get_demos(args, image_processor, model, tokenizer)
vti_vision, _ = obtain_visual_vti(
            model, input_images, rank=1
            )

visual_direction = vti_vision[1:]
```
```
vti_text, _ = obtain_textual_vti(
            model, input_ids, input_images, rank=1
            )
textual_direction = vti_text[1:]
```

2. Add the directions to the LVLM
```
add_vti_layers(model, torch.stack([textual_direction],dim=1).cuda(), alpha = [args.alpha_text])
```
Note that you need to specify the vision encoder of the model to add the visual direction

```
add_vti_layers(model.model.vision_tower.vision_tower.vision_model, torch.stack([visual_direction],dim=1).cuda(), alpha = [args.alpha_image])
```

## 🏅 Experiments

### MMHal-Bench [Download](https://llava-rlhf.github.io/)
```
python ./experiments/eval/run_mmhal_vti.py \
    --alpha_image 0.9 \
	--alpha_text 0.9 \
	--seed 42 \
	--image-folder dir/to/COCO/val2014/ \
	--data-file dir/to/COCO/ \
	--answers-file ./results/MMHal_answer.jsonl \
	--num_demos 70 \
	--mask_ratio 0.99 \
	--num_trials 50 
```

To evaluate
```
python experiments/eval/eval_mmhal.py \
	--response ./results/MMHal_answer.jsonl \
	--api-key YOUR OPENAI_API_KEY
```

### CHAIR
To be updated

### 📝 Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2410.11087):

```bibtex
@article{liu2024reducing,
  title={Reducing Hallucinations in Vision-Language Models via Latent Space Steering},
  author={Liu, Sheng and Ye, Haotian and Zou, James},
  journal={arXiv preprint arXiv:2410.15778},
  year={2024}
}
```