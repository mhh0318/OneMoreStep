import json

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.loaders import FromSingleFileMixin

from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import *
from diffusers.pipelines.pipeline_utils import _get_pipeline_class
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT

from diffusers_patch.models.unet_2d_condition_woct import UNet2DConditionWoCTModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_sub_model_oms(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    torch_dtype: torch.dtype,
    provider: Any,
    sess_options: Any,
    device_map: Optional[Union[Dict[str, torch.device], str]],
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]],
    offload_folder: Optional[Union[str, os.PathLike]],
    offload_state_dict: bool,
    model_variants: Dict[str, str],
    name: str,
    from_flax: bool,
    variant: str,
    low_cpu_mem_usage: bool,
    cached_folder: Union[str, os.PathLike],
):
    """Helper method to load the module `name` from `library_name` and `class_name`"""
    # retrieve class candidates
    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # retrive load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
            TRANSFORMERS_DUMMY_MODULES_FOLDER
        )
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    load_method = getattr(class_obj, load_method_name)

    # add kwargs to loading method
    import diffusers
    loading_kwargs = {}
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs["torch_dtype"] = torch_dtype
    if issubclass(class_obj, diffusers.OnnxRuntimeModel):
        loading_kwargs["provider"] = provider
        loading_kwargs["sess_options"] = sess_options

    is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)

    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )

    # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
    # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
    # This makes sure that the weights won't be initialized which significantly speeds up loading.
    if is_diffusers_model or is_transformers_model:
        loading_kwargs["device_map"] = device_map
        loading_kwargs["max_memory"] = max_memory
        loading_kwargs["offload_folder"] = offload_folder
        loading_kwargs["offload_state_dict"] = offload_state_dict
        loading_kwargs["variant"] = model_variants.pop(name, None)
        if from_flax:
            loading_kwargs["from_flax"] = True

        # the following can be deleted once the minimum required `transformers` version
        # is higher than 4.27
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            loading_kwargs.pop("variant")

        # if `from_flax` and model is transformer model, can currently not load with `low_cpu_mem_usage`
        if not (from_flax and is_transformers_model):
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            loading_kwargs["low_cpu_mem_usage"] = False
    # check if oms directory
    if 'oms' in name:
        config_name = os.path.join(cached_folder, name, 'config.json')
        with open(config_name, "r", encoding="utf-8") as f:
            index = json.load(f)
        file_path_or_name = index['_name_or_path']
        if 'subfolder' in index.keys():
            loading_kwargs["subfolder"] = index["subfolder"]
        loaded_sub_model = load_method(file_path_or_name, **loading_kwargs)
    else:
        # check if the module is in a subdirectory
        if os.path.isdir(os.path.join(cached_folder, name)):
            loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
        else:
            # else load from the root directory
            loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    return loaded_sub_model

class OMSPipeline(DiffusionPipeline, FromSingleFileMixin):

    def __init__(
        self,
        oms_module: UNet2DConditionWoCTModel,
        sd_pipeline: DiffusionPipeline,
        oms_text_encoder:Optional[CLIPTextModel],
        oms_tokenizer:Optional[CLIPTokenizer],
        sd_scheduler = None
    ):
        # assert sd_pipeline is not None

        if oms_tokenizer is None:
            oms_tokenizer = sd_pipeline.tokenizer
        if oms_text_encoder is None:
            oms_text_encoder = sd_pipeline.text_encoder

        self.is_dual_text_encoder = False  # For OMS with SDXL text encoders 

        self.register_modules(
            oms_module=oms_module,
            oms_text_encoder=oms_text_encoder,
            oms_tokenizer=oms_tokenizer,
            sd_pipeline = sd_pipeline
        )

        if sd_scheduler is None:
            self.scheduler = sd_pipeline.scheduler 
        else:
            self.scheduler = sd_scheduler
            sd_pipeline.scheduler = sd_scheduler

        self.vae_scale_factor = 2 ** (len(sd_pipeline.vae.config.block_out_channels) - 1)
        self.default_sample_size = sd_pipeline.unet.config.sample_size

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def oms_step(self, predict_v, latents, do_classifier_free_guidance_for_oms, oms_guidance_scale, generator, alpha_prod_t_prev):
        if do_classifier_free_guidance_for_oms:
            pred_uncond, pred_text = predict_v.chunk(2)
            predict_v = pred_uncond + oms_guidance_scale * (pred_text - pred_uncond)
        # so fking dirty but keep it for now
        alpha_prod_t = torch.zeros_like(alpha_prod_t_prev)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * predict_v
        # pred_original_sample = - predict_v
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        pred_prev_sample = pred_prev_sample
        # TODO unit variance but seem dont need it

        device = latents.device
        variance_noise = randn_tensor(
            latents.shape, generator=generator, device=device, dtype=latents.dtype
        )
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t 
        variance = torch.clamp(variance, min=1e-20) * variance_noise

        latents = pred_prev_sample + variance
        return latents

    def oms_text_encode(self, prompt, num_images_per_prompt, device):
        max_length = None if self.is_dual_text_encoder else self.oms_tokenizer.model_max_length
        if 'clip' in self.oms_text_encoder.config_class.model_type:
            tokenized_prompts = self.oms_tokenizer(prompt,
                                               padding='max_length',
                                               max_length=max_length,
                                               truncation=True,
                                               return_tensors='pt').input_ids
            if self.is_dual_text_encoder:
                tokenized_prompts = torch.stack([tokenized_prompts[0], tokenized_prompts[1]], dim=1)
            if self.is_dual_text_encoder:
                text_embeddings, _ = self.oms_text_encoder(
                    [tokenized_prompts[:, 0, :].to(device), tokenized_prompts[:, 1, :].to(device)])  # type: ignore
            else:
                text_embeddings = self.oms_text_encoder(tokenized_prompts.to(device))[0]  # type: ignore
        else: # T5
            tokenized_prompts = self.oms_tokenizer(prompt,
                                               padding='max_length',
                                               max_length=max_length,
                                               truncation=True,
                                               add_special_tokens=True,
                                               return_tensors='pt').input_ids
            # Note: t5 text encoder outputs "None" under fp16
            with torch.cuda.amp.autocast(dtype=torch.float32):
                text_embeddings = self.text_encoder(tokenized_prompts.to(device))[0] 

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return text_embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                from_flax=from_flax,
                use_safetensors=use_safetensors,
                custom_pipeline=custom_pipeline,
                custom_revision=custom_revision,
                variant=variant,
                load_connected_pipeline=load_connected_pipeline,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)

        # pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        # 2. Define which model components should load variants
        # We retrieve the information by matching whether variant
        # model checkpoints exist in the subfolders
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in config_dict
                variant_exists = is_folder and any(
                    p.split(".")[1].startswith(variant) for p in os.listdir(folder_path)
                )
                if variant_exists:
                    model_variants[folder] = variant

        # 3. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        pipeline_class = _get_pipeline_class(
            cls,
            config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline,
            cache_dir=cache_dir,
            revision=custom_revision,
        )

        # DEPRECATED: To be removed in 1.0.0
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
            version.parse(config_dict["_diffusers_version"]).base_version
        ) <= version.parse("0.5.1"):
            from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

            pipeline_class = StableDiffusionInpaintPipelineLegacy

            deprecation_message = (
                "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
                f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
                " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
                " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
                f" checkpoint {pretrained_model_name_or_path} to the format of"
                " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
                " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
            )
            deprecate("StableDiffusionInpaintPipelineLegacy", "1.0.0", deprecation_message, standard_warn=False)

        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs and make sure that optional component modules are filtered out
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Special case: safety_checker must be loaded separately when using `from_flax`
        if from_flax and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
            raise NotImplementedError(
                "The safety checker cannot be automatically loaded when loading weights `from_flax`."
                " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
                " separately if you need it."
            )

        # 5. Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # import it here to avoid circular import
        from diffusers import pipelines

        # 6. Load each module in the pipeline
        for name, (library_name, class_name) in logging.tqdm(init_dict.items(), desc="Loading pipeline components..."):
            # 6.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            class_name = class_name[4:] if class_name.startswith("Flax") else class_name

            # 6.2 Define all importable classes
            is_pipeline_module = hasattr(pipelines, library_name) 
            importable_classes = ALL_IMPORTABLE_CLASSES
            loaded_sub_model = None

            # 6.3 Use passed sub model or load class_name from library_name
            if name in passed_class_obj:
                # if the model is in a pipeline module, then we load it from the pipeline
                # check that passed_class_obj has correct parent class
                maybe_raise_or_warn(
                    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
                )

                loaded_sub_model = passed_class_obj[name]
            else:
                # load sub model
                loaded_sub_model = load_sub_model_oms(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=importable_classes,
                    pipelines=pipelines,
                    is_pipeline_module=is_pipeline_module,
                    pipeline_class=pipeline_class,
                    torch_dtype=torch_dtype,
                    provider=provider,
                    sess_options=sess_options,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    model_variants=model_variants,
                    name=name,
                    from_flax=from_flax,
                    variant=variant,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cached_folder=cached_folder,
                )
                logger.info(
                    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
                )

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        if pipeline_class._load_connected_pipes and os.path.isfile(os.path.join(cached_folder, "README.md")):
            modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
            connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for prefix in CONNECTED_PIPES_KEYS}
            load_kwargs = {
                "cache_dir": cache_dir,
                "resume_download": resume_download,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "use_auth_token": use_auth_token,
                "revision": revision,
                "torch_dtype": torch_dtype,
                "custom_pipeline": custom_pipeline,
                "custom_revision": custom_revision,
                "provider": provider,
                "sess_options": sess_options,
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "offload_state_dict": offload_state_dict,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "variant": variant,
                "use_safetensors": use_safetensors,
            }
            connected_pipes = {
                prefix: DiffusionPipeline.from_pretrained(repo_id, **load_kwargs.copy())
                for prefix, repo_id in connected_pipes.items()
                if repo_id is not None
            }

            for prefix, connected_pipe in connected_pipes.items():
                # add connected pipes to `init_kwargs` with <prefix>_<component_name>, e.g. "prior_text_encoder"
                init_kwargs.update(
                    {"_".join([prefix, name]): component for name, component in connected_pipe.components.items()}
                )

        # 7. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 8. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 9. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        return model

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        oms_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        oms_guidance_scale: float = 1.0,
        oms_flag: bool = True,
        **kwargs,
    ):
        """Pseudo-doc for OMS"""

        if oms_flag is True:
            if oms_prompt is not None:
                sd_prompt = prompt
                prompt = oms_prompt

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)


            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor
            device = self._execution_device
            ## Guidance flag for OMS
            if oms_guidance_scale is not None:
                do_classifier_free_guidance_for_oms = True
            else:
                do_classifier_free_guidance_for_oms = False


            oms_prompt_emb = self.oms_text_encode(prompt,num_images_per_prompt,device)
            if do_classifier_free_guidance_for_oms:
                oms_negative_prompt = ([''] * (batch_size // num_images_per_prompt))
                oms_negative_prompt_emb = self.oms_text_encode(oms_negative_prompt,num_images_per_prompt,device)

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.oms_module.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                oms_prompt_emb.dtype,
                device,
                generator,
                latents=None,
            )

            ## OMS CFG
            if do_classifier_free_guidance_for_oms:
                oms_prompt_emb = torch.cat([oms_negative_prompt_emb, oms_prompt_emb], dim=0)

        
            ## OMS to device
            oms_prompt_emb = oms_prompt_emb.to(device)


            ## Perform OMS
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            alpha_prod_t_prev =  alphas_cumprod[int(timesteps[0].item())] 
            latent_input_oms = torch.cat([latents] * 2) if do_classifier_free_guidance_for_oms else latents
            v_pred_oms = self.oms_module(latent_input_oms, oms_prompt_emb)['sample']
            latents = self.oms_step(v_pred_oms, latents, do_classifier_free_guidance_for_oms, oms_guidance_scale, generator, alpha_prod_t_prev)
            

            if oms_prompt is not None:
                prompt = sd_prompt

            print('OMS Completed')
        else:
            print("OMS unloaded")
            latents = None
        output = self.sd_pipeline(
                prompt = prompt,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                num_images_per_prompt = num_images_per_prompt,
                generator = generator,
                latents = latents,
                **kwargs
                )

        return output