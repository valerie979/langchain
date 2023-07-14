"""Run evaluator wrapper for string evaluators."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run, RunTypeEnum

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.load.serializable import Serializable
from langchain.schema import RUN_KEY, messages_from_dict
from langchain.schema.messages import BaseMessage, get_buffer_string


def _get_messages_from_run_dict(messages: List[dict]) -> List[BaseMessage]:
    if not messages:
        return []
    first_message = messages[0]
    if "lc" in first_message:
        return [loads(dumps(message)) for message in messages]
    else:
        return messages_from_dict(messages)


class StringRunMapper(Serializable):
    """Extract items to evaluate from the run object."""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["prediction", "input"]

    @abstractmethod
    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""

    def __call__(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        return self.map(run)


class LLMStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object."""

    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                chat_messages = _get_messages_from_run_dict(messages)
            elif isinstance(messages[0], list):
                # Runs from Tracer have messages as a list of lists of dicts
                chat_messages = _get_messages_from_run_dict(messages[0])
            else:
                raise ValueError(f"Could not extract messages to evaluate {messages}")
            return get_buffer_string(chat_messages)
        raise ValueError(f"Could not extract messages to evaluate {messages}")

    def serialize_inputs(self, inputs: Dict) -> str:
        if "prompts" in inputs:  # Should we even accept this?
            input_ = "\n\n".join(inputs["prompts"])
        elif "prompt" in inputs:
            input_ = inputs["prompt"]
        elif "messages" in inputs:
            input_ = self.serialize_chat_messages(inputs["messages"])
        else:
            raise ValueError("LLM Run must have either messages or prompts as inputs.")
        return input_

    def serialize_outputs(self, outputs: Dict) -> str:
        if not outputs.get("generations"):
            raise ValueError("Cannot evaluate LLM Run without generations.")
        generations: List[Dict] = outputs["generations"]
        if not generations:
            raise ValueError("Cannot evaluate LLM run with empty generations.")
        first_generation: Dict = generations[0]
        if isinstance(first_generation, list):
            # Runs from Tracer have generations as a list of lists of dicts
            # Whereas Runs from the API have a list of dicts
            first_generation = first_generation[0]
        if "message" in first_generation:
            output_ = self.serialize_chat_messages([first_generation["message"]])
        else:
            output_ = first_generation["text"]
        return output_

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if run.run_type != "llm":
            raise ValueError("LLM RunMapper only supports LLM runs.")
        elif not run.outputs:
            if run.error:
                raise ValueError(
                    f"Cannot evaluate errored LLM run {run.id}: {run.error}"
                )
            else:
                raise ValueError(
                    f"Run {run.id} has no outputs. Cannot evaluate this run."
                )
        else:
            try:
                inputs = self.serialize_inputs(run.inputs)
            except Exception as e:
                raise ValueError(
                    f"Could not parse LM input from run inputs {run.inputs}"
                ) from e
            try:
                output_ = self.serialize_outputs(run.outputs)
            except Exception as e:
                raise ValueError(
                    f"Could not parse LM prediction from run outputs {run.outputs}"
                ) from e
            return {"input": inputs, "prediction": output_}


class ChainStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object from a chain."""

    input_key: Optional[str] = None
    """The key from the model Run's inputs to use as the eval input."""
    prediction_key: Optional[str] = None
    """The key from the model Run's outputs to use as the eval prediction."""

    def _get_key(self, source: Dict, key: Optional[str], which: str) -> str:
        if key is not None:
            return source[key]
        elif len(source) == 1:
            return next(iter(source.values()))
        else:
            raise ValueError(
                f"Could not map run {which} with multiple keys: "
                f"{source}\nPlease manually specify a {which}_key"
            )

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        if run.run_type != "chain":
            raise ValueError("Chain RunMapper only supports Chain runs.")
        if self.input_key not in run.inputs:
            raise ValueError(f"Run {run.id} does not have input key {self.input_key}.")
        elif self.prediction_key not in run.outputs:
            raise ValueError(
                f"Run {run.id} does not have prediction key {self.prediction_key}."
            )
        else:
            input_ = self._get_key(run.inputs, self.input_key, "input")
            prediction = self._get_key(run.outputs, self.prediction_key, "prediction")
            return {
                "input": input_,
                "prediction": prediction,
            }


class ToolStringRunMapper(StringRunMapper):
    """Map an input to the tool."""

    def map(self, run: Run) -> Dict[str, str]:
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        return {"input": run.inputs["input"], "prediction": run.outputs["output"]}


class AutoStringRunMapper(StringRunMapper):
    """Automatically select the appropriate StringRunMapper based on the run type."""

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if run.run_type is None:
            # Call the superclass's map method
            return super().map(run)
        elif run.run_type == RunTypeEnum.llm:
            # Use LLMStringRunMapper for LLM runs
            mapper = LLMStringRunMapper()
        elif run.run_type == RunTypeEnum.chain:
            # Use ChainStringRunMapper for Chain runs
            mapper = ChainStringRunMapper()
        elif run.run_type == RunTypeEnum.tool:
            # Use ToolStringRunMapper for Tool runs
            mapper = ToolStringRunMapper()
        else:
            raise ValueError(f"Unsupported run type: {run.run_type}")
        
        return mapper.map(run)


class StringExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""

    reference_key: Optional[str] = None

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["reference"]

    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        chat_messages = _get_messages_from_run_dict(messages)
        return get_buffer_string(chat_messages)

    def map(self, example: Example) -> Dict[str, str]:
        """Maps the Example, or dataset row to a dictionary."""
        if not example.outputs:
            raise ValueError(
                f"Example {example.id} has no outputs to use as a reference."
            )
        if self.reference_key is None:
            if len(example.outputs) > 1:
                raise ValueError(
                    f"Example {example.id} has multiple outputs, so you must"
                    " specify a reference_key."
                )
            else:
                output = list(example.outputs.values())[0]
                return {
                    "reference": self.serialize_chat_messages([output])
                    if isinstance(output, dict)
                    and output.get("type")
                    and output.get("data")
                    else output
                }
        elif self.reference_key not in example.outputs:
            raise ValueError(
                f"Example {example.id} does not have reference key"
                f" {self.reference_key}."
            )
        return {"reference": example.outputs[self.reference_key]}

    def __call__(self, example: Example) -> Dict[str, str]:
        """Maps the Run and Example to a dictionary."""
        if not example.outputs:
            raise ValueError(
                f"Example {example.id} has no outputs to use as areference label."
            )
        return self.map(example)


class StringRunEvaluatorChain(Chain, RunEvaluator):
    """Evaluate Run and optional examples."""

    run_mapper: Union[AutoStringRunMapper, LLMStringRunMapper, ChainStringRunMapper, ToolStringRunMapper]
    """Maps the Run to a dictionary with 'input' and 'prediction' strings."""
    example_mapper: Optional[StringExampleMapper] = None
    """Maps the Example (dataset row) to a dictionary
    with a 'reference' string."""
    name: str
    """The name of the evaluation metric."""
    string_evaluator: StringEvaluator
    """The evaluation chain."""

    @classmethod
    def from_run_and_data_type(
        cls,
        evaluator: StringEvaluator,
        run_type: Optional[str],
        data_type: DataType,
        input_key: Optional[str] = None,
        prediction_key: Optional[str] = None,
        reference_key: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> StringRunEvaluatorChain:
        """Create a StringRunEvaluatorChain from a StringEvaluator, run_type, and data_type."""
        if run_type is None:
            run_mapper = AutoStringRunMapper()
        elif run_type == RunTypeEnum.llm:
            run_mapper = LLMStringRunMapper()
        elif run_type == RunTypeEnum.chain:
            run_mapper = ChainStringRunMapper()
        elif run_type == RunTypeEnum.tool:
            run_mapper = ToolStringRunMapper()
        else:
            raise ValueError(f"Unsupported run type: {run_type}")

        return cls(
            run_mapper=run_mapper,
            example_mapper=StringExampleMapper(reference_key=reference_key),
            name=evaluator.name,
            string_evaluator=evaluator,
            data_type=data_type,
            input_key=input_key,
            prediction_key=prediction_key,
            tags=tags,
        )