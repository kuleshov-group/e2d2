import inspect
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import attr
import hydra.utils
import torch
from transformers import (
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
    LogitsProcessorList,
    PretrainedConfig,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import ModelOutput

# Add the local directory (enables hydra.utils.instantiate for local imports)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

# Local imports not used, but added here so that HF push_to_hub adds them to model repo
# noinspection PyUnresolvedReferences
from src.backbone.automodel import AutoModelFromPreTrained  # noqa: F401
from src.backbone.encoder_decoder import LLMasEncoderDecoder  # noqa: F401
from src.noise_schedule.noise_schedules import (  # noqa: F401
    CosineNoise,
    ExponentialNoise,
    LinearNoise,
    LogarithmicNoise,
)


@dataclass
class DenoiserInput(OrderedDict):
    """Input to the denoiser model."""

    xt: torch.LongTensor  # (B, L) token_ids
    x0: Optional[torch.LongTensor] = None  # (B, L) token_ids (not used in gen.)
    attention_mask: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Union[torch.FloatTensor, Cache]] = None
    context_mask: Optional[torch.FloatTensor] = None
    tokens_mask: Optional[torch.FloatTensor] = None  # (B, L)
    t: Optional[torch.FloatTensor] = None  # (B,) | # (B, L)
    alpha_t: Optional[torch.FloatTensor] = None  # (B,) | (B, 1|L) | (B, 1|L, 1)
    alpha_t_prime: Optional[torch.FloatTensor] = None  # (B,) | (B, 1|L) | (B, 1|L, 1)
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossAndNllOutput(OrderedDict):
    """Loss output for denoiser models."""

    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    other_loss_terms: dict = field(default_factory=dict)


@attr.s(auto_attribs=True, init=False)
class DenoiserOutput(OrderedDict):
    """Output of the denoiser model."""

    denoiser_output: torch.FloatTensor
    logits: Optional[torch.FloatTensor] = None
    tokens_mask: Optional[torch.FloatTensor] = None  # Which tokens contribute to loss
    past_key_values: Optional[Cache] = None
    loss: Optional[torch.FloatTensor] = None
    nlls: Optional[torch.FloatTensor] = None

    def __init__(
        self,
        denoiser_output: torch.FloatTensor,
        logits: Optional[torch.FloatTensor] = None,
        tokens_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        loss: Optional[torch.FloatTensor] = None,
        nlls: Optional[torch.FloatTensor] = None,
        **output_kwargs: Any,
    ):
        super().__init__()
        self["denoiser_output"] = denoiser_output
        self["logits"] = logits
        self["tokens_mask"] = tokens_mask
        self["past_key_values"] = past_key_values
        self["loss"] = loss
        self["nlls"] = nlls
        for k, v in output_kwargs.items():
            self[k] = v
        for k, v in self.items():
            setattr(self, k, v)


class DenoiserConfig(PretrainedConfig):
    """Configuration class for Denoiser models.

    This class is used to initialize the model and contains all the necessary
    parameters for the model's architecture.
    """

    model_type = "denoiser"

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        time_conditioned_backbone: bool | None = None,
        attn_backend: str = "sdpa",  # "sdpa", "flash_attention_2", "flex_attention"
        **kwargs,
    ):
        super().__init__(**kwargs)
        for v in [
            "vocab_size",
            "mask_token_id",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "pad_vocab_size_multiple",
        ]:
            if tokenization_config is not None and (
                getattr(self, v, None) is None or v in tokenization_config
            ):
                setattr(self, v, tokenization_config.get(v, None))
            else:
                setattr(self, v, None)
        self.backbone_config = backbone_config
        self.noise_config = noise_config
        self.tokenization_config = tokenization_config
        self.length = length
        self.time_conditioned_backbone = time_conditioned_backbone
        self.attn_backend = attn_backend


class Denoiser(ABC, PreTrainedModel):
    """Abstract base class for denoising models.

    This class defines the interface for AR, Diffusion, and Flow-based parametrizations.
    """

    config_class = DenoiserConfig

    def __init__(
        self,
        config: DenoiserConfig,
        **kwargs,
    ):
        """
        Initialize the Denoiser with a configuration and optional dataset type.

        Parameters:
            config (Any): Configuration object for the model.
        """
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.backbone = hydra.utils.instantiate(config.backbone_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=True,
        )
        self.noise_schedule = (
            hydra.utils.instantiate(config.noise_config)
            if config.noise_config is not None
            else None
        )
        self.time_conditioned_backbone = (
            config.time_conditioned_backbone
            if config.time_conditioned_backbone is not None
            else "noise" in inspect.getfullargspec(self.backbone.forward).args
        )

    @abstractmethod
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
    ) -> DenoiserInput:
        """
        Prepare inputs for the model.

        Parameters:
            input_ids (LongTensor): Input tensor to the model.
            attention_mask (Optional[FloatTensor]): Attention mask for the model.
            t (Optional[FloatTensor]): Time step for the model.
            past_key_values (Optional[Cache]): Past key values for the model.
        Returns:
            Denoiser inputs.
        """
        raise NotImplementedError("Denoiser subclasses must implement _prepare_inputs")

    @abstractmethod
    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        """
        Compute the loss for the denoising model.

        Parameters:
            model_output (FloatTensor): Output tensor from self.forward.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            LossAndNllOutput: loss (FloatTensor) and nlls (FloatTensor).
        """
        raise NotImplementedError("Denoiser subclasses must implement _compute_loss")

    def _forward(
        self,
        backbone_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.FloatTensor:
        """
        Forward pass for the denoiser model returns probabilities over denoised
        sequence.

        Some classes may need to override this method.

        Parameters:
            backbone_output (FloatTensor): Output tensor from the backbone model.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Model outputs (FloatTensor).
        """
        return torch.log_softmax(backbone_output, dim=-1)  # type: ignore

    def _backbone_forward(
        self,
        denoiser_inputs: DenoiserInput,
        **backbone_kwargs: Any,
    ) -> ModelOutput:
        """Forward pass for the backbone model (should return logits).

        Some classes may need to override this method.

        Parameters:
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.
            return_updated_cache (bool): If True, return past_key_values instead of
                logits.

        Returns:
            Backbone output (ModelOutput instance).
        """
        if self.time_conditioned_backbone:
            return self.backbone(
                denoiser_inputs.xt,
                attention_mask=denoiser_inputs.attention_mask,
                past_key_values=denoiser_inputs.past_key_values,
                noise=denoiser_inputs.alpha_t,
                **denoiser_inputs.backbone_kwargs,
                **backbone_kwargs,
            )
        return self.backbone(
            denoiser_inputs.xt,
            attention_mask=denoiser_inputs.attention_mask,
            past_key_values=denoiser_inputs.past_key_values,
            **denoiser_inputs.backbone_kwargs,
            **backbone_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        compute_loss: bool | None = True,
        **kwargs,
    ) -> DenoiserOutput:
        """
        Perform a forward pass through the denoising model and
        (optionally) compute the loss.

        Parameters:
            input_ids (LongTensor): Input tensor to the model.
            attention_mask (Optional[FloatTensor]): Attention mask for the model.
            context_mask (Optional[FloatTensor]): Indicator for context tokens.
            t (Optional[FloatTensor]): Denoising time step for the model.
            past_key_values (Optional[Cache]): KV cache.
            compute_loss (Optional[bool]): Flag to compute loss.

        Returns:
            DenoiserOutput
        """
        denoiser_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            past_key_values=past_key_values,
            t=t,
        )

        backbone_output = self._backbone_forward(denoiser_inputs, **kwargs)
        new_past_key_values = getattr(backbone_output, "past_key_values", None)
        backbone_output = getattr(backbone_output, "logits", backbone_output[0])
        denoiser_output = self._forward(
            backbone_output,
            denoiser_inputs,
            **kwargs,
        )

        if compute_loss:
            loss_and_nll = self._compute_loss(
                model_output=denoiser_output, denoiser_inputs=denoiser_inputs, **kwargs
            )
            loss = loss_and_nll.loss
            nlls = loss_and_nll.nlls
            other_loss_terms = loss_and_nll.other_loss_terms
        else:
            loss, nlls = None, None
            other_loss_terms = {}

        return DenoiserOutput(
            denoiser_output=denoiser_output,
            logits=backbone_output,
            past_key_values=new_past_key_values,
            tokens_mask=denoiser_inputs.tokens_mask,
            loss=loss,
            nlls=nlls,
            **other_loss_terms,
        )

    @staticmethod
    def _sample_categorical(categorical_probs, do_sample=True):
        """Helper function to sample from a categorical distribution."""
        # TODO: for greedy, can we skip fp64 casting?
        categorical_probs = categorical_probs.to(torch.float64)
        if not do_sample:
            return categorical_probs.argmax(dim=-1)
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(
            categorical_probs.dtype
        )
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    @staticmethod
    def _preprocess_attention_mask(attention_mask, dtype):
        min_dtype = torch.finfo(dtype).min
        attention_mask = torch.where(
            (attention_mask == 0.0).bool(),  # type: ignore
            min_dtype,
            0.0,
        ).to(dtype)
        return attention_mask

    def _prepare_inputs_inference(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        context: torch.LongTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        cache: Dict[str, Any] | None = None,
        **backbone_kwargs: Any,
    ) -> Tuple[DenoiserInput, Dict[str, Any]]:
        assert input_ids is not None or context is not None, (
            "Must provide either input_ids or context."
        )
        cache = cache if cache is not None else {}
        past_key_values = cache.pop("past_key_values", DynamicCache())
        if context is not None:
            if input_ids is not None:
                if context_mask is None:
                    context_mask = torch.cat(
                        [torch.ones_like(context), torch.zeros_like(input_ids)], dim=-1
                    )
                input_ids = torch.cat([context, input_ids], dim=-1)
            else:
                input_ids = context
                context_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            cache_length = self._get_past_key_values_seq_length(past_key_values)
            full_seq_length = cache_length + input_ids.shape[-1]
            attention_mask = torch.ones(
                (input_ids.shape[0], 1, input_ids.shape[1], full_seq_length),
                device=input_ids.device,
            )  # Make attention mask 4D
            attention_mask = self._preprocess_attention_mask(
                attention_mask, dtype=torch.float
            )
        return DenoiserInput(
            xt=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            context_mask=context_mask,
            backbone_kwargs=backbone_kwargs,
        ), cache

    @staticmethod
    def _get_past_key_values_seq_length(past_key_values: DynamicCache):
        seq_length = 0
        for i in range(len(past_key_values)):
            if past_key_values[i][0].shape[0] > 0:  # type: ignore
                seq_length = max(
                    past_key_values[i][0].shape[-2],  # type: ignore
                    seq_length,
                )
        return seq_length

    def update_cache(
        self,
        inputs: torch.LongTensor,
        cache: Dict[str, Any] | None = None,
        **backbone_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Cache the key-value pairs for the context.
        Args:
            inputs (torch.LongTensor): The context tensor.
            cache (Dict[str, Any | None): Cache objects, e.g., past_key_values.
        Returns:
            Dict: Updated cache objects, e.g., past_key_values.
        """
        context_input, cache = self._prepare_inputs_inference(
            input_ids=inputs, cache=cache, return_updated_cache=True, **backbone_kwargs
        )
        backbone_output = self._backbone_forward(
            context_input,
            return_updated_cache=True,  # Will get absorbed in backbone_kwargs
            **cache,
        )
        backbone_output = {k: v for k, v in backbone_output.items()}
        backbone_output.pop("logits", None)  # Do not store logits in cache
        cache = cache | backbone_output
        return cache

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> GenerateOutput | torch.LongTensor:
        """Generates sample from denoising model.
        Follows signature of transformers.GenerationMixin.
        """
        raise NotImplementedError("Denoiser subclasses must implement generate")
