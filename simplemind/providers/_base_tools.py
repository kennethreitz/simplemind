import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Literal, get_origin

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType


def _is_literal(t: Any) -> bool:
    return get_origin(t) is Literal


def _is_required(field, func_signature, arg_name) -> bool:
    param = func_signature.parameters[arg_name]
    # If parameter has a default value that's not a FieldInfo, it's not required
    if param.default is not inspect.Parameter.empty and not isinstance(
        param.default, FieldInfo
    ):
        return False
    # If the field has a default that's not undefined, it's not required
    return isinstance(field.default, PydanticUndefinedType)


class BaseToolConfig(BaseModel):
    TYPE_CONVERSION: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }


class BaseToolProperty(BaseModel):
    type: str = Field(serialization_alias="type_")
    enum: list[str] | None = None
    description: str


class BaseTool(BaseModel, ABC):
    name: str
    description: str
    properties: dict[str, BaseToolProperty]
    required: list[str] | None = None
    config: ClassVar[BaseToolConfig] = BaseToolConfig()
    raw_func: Any | None = None
    tool_id: str | None = None
    function_result: str | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self.raw_func is not None
        return self.raw_func(*args, **kwargs)

    def is_executed(self) -> bool:
        return self.function_result is not None

    def reset_result(self) -> None:
        self.function_result = None

    @classmethod
    def convert_type(cls, field_type) -> str:
        if _is_literal(field_type):
            return cls.config.TYPE_CONVERSION[str]

        field_type_converted = cls.config.TYPE_CONVERSION.get(field_type, None)

        if field_type_converted is None:
            raise TypeError(f"Field of type {field_type} is not supported")

        return field_type_converted

    def get_properties_schema(self, **kwargs) -> dict[str, dict]:
        new_kwargs: dict = {"exclude_none": True} | kwargs
        return {
            k: v.model_dump(**new_kwargs) for k, v in self.properties.items()
        }

    @classmethod
    def from_function(cls, func: Callable | "BaseTool"):
        # Check if the func passed is an instace of BaseTool
        if hasattr(func, "raw_func"):
            return func

        annotations = getattr(func, "__annotations__", {})
        properties = {}
        required = []
        enum_values = None
        func_signature = inspect.signature(func)

        for n, (arg_name, arg_type) in enumerate(annotations.items()):
            if (  # Skipping 'return' annotation (i.e.```-> str```)
                arg_name != "return"
            ):
                # Check if argument has metadata (from Annotated)
                if hasattr(arg_type, "__metadata__"):
                    field = arg_type.__metadata__[
                        0
                    ]  # Get Field info from metadata
                    field_type = arg_type.__origin__  # Get actual type
                # Check if argument has a default value in signature
                elif (
                    sig_param := func_signature.parameters[arg_name]
                ).default is not inspect.Parameter.empty:
                    field = sig_param.default  # Use default as Field
                    field_type = arg_type  # Use plain type annotation
                else:
                    # Raise error if no Field annotation found
                    raise ValueError(
                        f"Please add a Field annotation to `{func.__name__}.{arg_name}` parameter"
                    )

                field_type_converted = cls.convert_type(field_type)

                if _is_literal(field_type):
                    enum_values = [str(x) for x in field_type.__args__]

                properties[arg_name] = BaseToolProperty(
                    type=field_type_converted,
                    description=field.description,
                    enum=enum_values,
                )
                if _is_required(field, func_signature, arg_name):
                    required.append(arg_name)

        return cls(
            name=func.__name__,
            description=(func.__doc__ or "").strip(),
            properties=properties,
            required=required,
            raw_func=func,
        )

    @abstractmethod
    def get_input_schema(self) -> Any: ...

    @abstractmethod
    def handle(self, message) -> None: ...

    @abstractmethod
    def get_response_schema(self) -> Any: ...
