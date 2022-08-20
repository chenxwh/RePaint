# run "pip install --upgrade gdown && bash ./download.sh" beforehand

import os
import torch as th
import torch.nn.functional as F
import shutil
import tempfile
import random
import yaml
from PIL import Image
from cog import BasePredictor, Path, Input, BaseModel

import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)


class Output(BaseModel):
    mask: Path
    masked_image: Path
    inpainted_image: Path


class Predictor(BasePredictor):
    def setup(self):
        self.face_mask_conf = "confs/face_example.yml"
        self.conf_path = {
            "CelebA-HQ": {
                "thick": "confs/test_c256_thick.yml",
                "thin": "confs/test_c256_thin.yml",
                "every second line": "confs/test_c256_ev2li.yml",
                "super-resolution": "confs/test_c256_nn2.yml",
                "expand": "confs/test_c256_ex64.yml",
                "half": "confs/test_c256_genhalf.yml",
            },
            "ImageNet": {
                "thick": "confs/test_inet256_thick.yml",
                "thin": "confs/test_inet256_thin.yml",
                "every second line": "confs/test_inet256_ev2li.yml",
                "super-resolution": "confs/test_inet256_nn2.yml",
                "expand": "confs/test_inet256_ex64.yml",
                "half": "confs/test_inet256_genhalf.yml",
            },
            "Places": {
                "thick": "confs/test_p256_thick.yml",
                "thin": "confs/test_p256_thin.yml",
                "every second line": "confs/test_p256_ev2li.yml",
                "super-resolution": "confs/test_p256_nn2.yml",
                "expand": "confs/test_p256_ex64.yml",
                "half": "confs/test_p256_genhalf.yml",
            },
        }

        self.mask_type_dir = {
            "face mask": "data/datasets/gt_keep_masks/face",
            "thick": "data/datasets/gt_keep_masks/thick",
            "thin": "data/datasets/gt_keep_masks/thin",
            "every second line": "data/datasets/gt_keep_masks/ev2li",
            "super-resolution": "data/datasets/gt_keep_masks/nn2",
            "expand": "data/datasets/gt_keep_masks/ex64",
            "half": "data/datasets/gt_keep_masks/genhalf",
        }

    def predict(
        self,
        image: Path = Input(
            description="Input image. Facial images are expected to be aligned. If not, you can use https://replicate.com/cjwbw/face-align-cog to align your image first.",
        ),
        model: str = Input(
            default="CelebA-HQ",
            description="Choose a model depending on the input image.",
            choices=[
                "CelebA-HQ",
                "ImageNet",
                "Places",
            ],
        ),
        mask: str = Input(
            default="face mask",
            description="Choose a type for masking the image before repainting. Please refer to the Examples to see what each type of the mask looks like. ",
            choices=[
                "face mask",
                "thick",
                "thin",
                "every second line",
                "super-resolution",
                "expand",
                "half",
            ],
        ),
    ) -> Output:
        if not model == "CelebA-HQ":
            assert (
                not mask == "face mask"
            ), f"face mask is not available for {model} model"

        conf_path = (
            self.face_mask_conf if mask == "face mask" else self.conf_path[model][mask]
        )
        conf = conf_mgt.conf_base.Default_Conf()
        conf.update(yamlread(conf_path))

        # print(conf)
        if model == "CelebA-HQ":
            conf[
                "model_path"
            ] = "./data/pretrained/celeba256_250000.pt"  # not all confs.yml set to this path

        # print(conf)

        device = "cuda:0"
        model, diffusion = create_model_and_diffusion(
            **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
        )
        model.load_state_dict(
            dist_util.load_state_dict(
                os.path.expanduser(conf.model_path), map_location="cpu"
            )
        )
        model.to(device)
        if conf.use_fp16:
            model.convert_to_fp16()
        model.eval()

        show_progress = conf.show_progress

        if conf.classifier_scale > 0 and conf.classifier_path:
            print("loading classifier...")
            classifier = create_classifier(
                **select_args(conf, classifier_defaults().keys())
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(
                    os.path.expanduser(conf.classifier_path), map_location="cpu"
                )
            )

            classifier.to(device)
            if conf.classifier_use_fp16:
                classifier.convert_to_fp16()
            classifier.eval()

            def cond_fn(x, t, y=None, gt=None, **kwargs):
                assert y is not None
                with th.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(x_in, t)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    return (
                        th.autograd.grad(selected.sum(), x_in)[0]
                        * conf.classifier_scale
                    )

        else:
            cond_fn = None

        def model_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            return model(x, t, y if conf.class_cond else None, gt=gt)

        print("sampling...")

        dset = "eval"
        eval_name = conf.get_default_eval_name()

        # overwrite image paths with customised input
        gt_dir, mask_dir = "gt_dir", "mask_dir"
        for d in [gt_dir, mask_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        print(str(image))

        # shutil.copy(str(image), os.path.join(gt_dir, "input.png"))
        shutil.copy(str(image), gt_dir)

        mask_type_dir = self.mask_type_dir[mask]
        # randomly select one mask from the mask type
        idx = random.randint(0, len(os.listdir(mask_type_dir)) - 1)
        shutil.copy(
            os.path.join(mask_type_dir, os.listdir(mask_type_dir)[idx]),
            os.path.join(mask_dir, "mask.png"),
        ) 
        # shutil.copy(f'{mask_type_dir}/{os.listdir(mask_type_dir)[idx]}',
        #     mask_dir
        # )
        print(f'{mask_type_dir}/{os.listdir(mask_type_dir)[idx]}')

        conf["data"]["eval"][eval_name]["gt_path"] = gt_dir
        conf["data"]["eval"][eval_name]["mask_path"] = mask_dir

        dl = conf.get_dataloader(dset=dset, dsName=eval_name)
        batch = next(iter(dl))

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {"gt": batch["GT"]}

        gt_keep_mask = batch.get("gt_keep_mask")
        if gt_keep_mask is not None:
            model_kwargs["gt_keep_mask"] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf,
        )
        srs = toU8(result["sample"])
        lrs = toU8(
            result.get("gt") * model_kwargs.get("gt_keep_mask")
            + (-1)
            * th.ones_like(result.get("gt"))
            * (1 - model_kwargs.get("gt_keep_mask"))
        )
        gt_keep_masks = toU8((model_kwargs.get("gt_keep_mask") * 2 - 1))

        print("sampling complete")

        mask_path = Path(tempfile.mkdtemp()) / "mask.png"
        masked_image_path = Path(tempfile.mkdtemp()) / "masked_image.png"
        inpainted_image_path = Path(tempfile.mkdtemp()) / "inpainted_image.png"

        Image.fromarray(gt_keep_masks[0]).save(str(mask_path))
        Image.fromarray(lrs[0]).save(str(masked_image_path))
        Image.fromarray(srs[0]).save(str(inpainted_image_path))

        Image.fromarray(gt_keep_masks[0]).save("a.png")
        Image.fromarray(lrs[0]).save("b.png")
        Image.fromarray(srs[0]).save("c.png")

        return Output(
            mask=mask_path,
            masked_image=masked_image_path,
            inpainted_image=inpainted_image_path,
        )


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample
