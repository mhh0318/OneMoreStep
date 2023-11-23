import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class SDXLTextEncoder(torch.nn.Module):
    """Wrapper around HuggingFace text encoders for SDXL.

    Creates two text encoders (a CLIPTextModel and CLIPTextModelWithProjection) that behave like one.

    Args:
        model_name (str): Name of the model's text encoders to load. Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        encode_latents_in_fp16 (bool): Whether to encode latents in fp16. Defaults to True.
    """

    def __init__(self, model_name='stabilityai/stable-diffusion-xl-base-1.0', encode_latents_in_fp16=True, torch_dtype=None):
        super().__init__()
        if torch_dtype is None:
            torch_dtype = torch.float16 if encode_latents_in_fp16 else None
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch_dtype)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name,
                                                                          subfolder='text_encoder_2',
                                                                          torch_dtype=torch_dtype)

    @property
    def device(self):
        return self.text_encoder.device

    def forward(self, tokenized_text):
        # first text encoder
        conditioning = self.text_encoder(tokenized_text[0], output_hidden_states=True).hidden_states[-2]
        # second text encoder
        text_encoder_2_out = self.text_encoder_2(tokenized_text[1], output_hidden_states=True)
        pooled_conditioning = text_encoder_2_out[0]  # (batch_size, 1280)
        conditioning_2 = text_encoder_2_out.hidden_states[-2]  # (batch_size, 77, 1280)

        conditioning = torch.concat([conditioning, conditioning_2], dim=-1)
        return conditioning, pooled_conditioning


class SDXLTokenizer:
    """Wrapper around HuggingFace tokenizers for SDXL.

    Tokenizes prompt with two tokenizers and returns the joined output.

    Args:
        model_name (str): Name of the model's text encoders to load. Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
    """

    def __init__(self, model_name='stabilityai/stable-diffusion-xl-base-1.0'):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer_2')

    def __call__(self, prompt, padding, truncation, return_tensors, max_length=None):
        tokenized_output = self.tokenizer(
            prompt,
            padding=padding,
            max_length=self.tokenizer.model_max_length if max_length is None else max_length,
            truncation=truncation,
            return_tensors=return_tensors)
        tokenized_output_2 = self.tokenizer_2(
            prompt,
            padding=padding,
            max_length=self.tokenizer_2.model_max_length if max_length is None else max_length,
            truncation=truncation,
            return_tensors=return_tensors)

        # Add second tokenizer output to first tokenizer
        for key in tokenized_output.keys():
            tokenized_output[key] = [tokenized_output[key], tokenized_output_2[key]]
        return tokenized_output
