# -*- coding: utf-8 -*-

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    _get_protocol_attrs,
)

import attr
import numpy as np
import pandas as pd

TUnion = Union[Type[Any], Tuple[Type[Any], ...]]


def _isprotocol_subclass(cls: Any, protocol: Type[Any]) -> bool:
    fields = set(attr.fields_dict(cls).keys())
    meths = set(key for key in cls.__dict__.keys() if not key.startswith("_"))
    fm = fields | meths
    ret = all([attr in fm for attr in _get_protocol_attrs(protocol)])

    return ret


def isprotocol_subclass(cls: Any, class_or_tuple: TUnion) -> bool:
    """Return True if cls is a subclass of class or tuple of classes.

    Parameters
    ----------
    cls : Any
        Attr based Dataclass
    class_or_tuple : Union[Type[Any], Tuple[Type[Any], ...]]
        Class or tuple of classes

    Returns
    -------
    bool
        True if cls is a subclass of class or one of classes in tuple.

    """
    assert attr.has(cls), f"cls: {cls} is not an attr based dataclass."
    if isinstance(class_or_tuple, (tuple,)):
        return any([_isprotocol_subclass(cls, val) for val in class_or_tuple])

    return _isprotocol_subclass(cls, class_or_tuple)


def _isdataprotocol_subclass(cls: Type[Any], protocol: Type[Any]) -> bool:
    fields = set(attr.fields_dict(cls).keys())

    attrs = set()
    for base in cls.__mro__[:-1]:  # without object
        if base.__name__ in ("Protocol", "Generic"):
            continue
        annotations = getattr(base, "__annotations__", {})
        for a in list(annotations.keys()):
            if not a.startswith("_abc_") and a not in (
                "__abstractmethods__",
                "__annotations__",
                "__weakref__",
                "_is_protocol",
                "_is_runtime_protocol",
                "__dict__",
                "__args__",
                "__slots__",
                "__next_in_mro__",
                "__parameters__",
                "__origin__",
                "__orig_bases__",
                "__extra__",
                "__tree_hash__",
                "__doc__",
                "__subclasshook__",
                "__init__",
                "__new__",
                "__module__",
                "_MutableMapping__marker",
                "_gorg",
            ):
                attrs.add(a)

    ret = all([a in fields for a in attrs])

    return ret


def isdataprotocol_subclass(cls: Type[Any], class_or_tuple: TUnion) -> bool:
    """Return True if cls is a subclass of class or tuple of classes.

    Parameters
    ----------
    cls : Any
        Attr based Dataclass
    class_or_tuple : Union[Type[Any], Tuple[Type[Any], ...]]
        Class or tuple of classes

    Returns
    -------
    bool
        True if cls is data protocol subclass of class
        or one of classes in tuple.

    """
    assert attr.has(cls), f"cls: {cls} is not an attr based dataclass."
    if isinstance(class_or_tuple, (tuple,)):
        return any(
            [_isdataprotocol_subclass(cls, val) for val in class_or_tuple]
        )

    return _isprotocol_subclass(cls, class_or_tuple)


def dataframe_to_dataclass(
    name: str,
    df: pd.DataFrame,
    min_itemsize: Optional[Union[Dict[str, int], int]] = None,
) -> Type[Any]:
    """Create Dataclass type from DataFrame.

    Parameters
    ----------
    name : str
        Name of created Dataclass type
    df : pd.DataFrame
        DataFrame to create Dataclass from
    min_itemsize : Optional[Union[Dict[str, int], int]], optional
        min_itemsize for string columns, by default None

    Returns
    -------
    Type[Any]
        Dataclass type

    Raises
    ------
    TypeError
        For an unsupported type stored in DataFrame
    ValueError
        For an empty string column without specified min_itemsize

    """
    attrs = {}
    for key, t in df.dtypes.items():
        if issubclass(t.type, np.number):
            attrs[key] = attr.ib(type=t.type)
        elif issubclass(t.type, np.object_):
            # Object is a str or bytes
            if isinstance(df.loc[0, key], (str, bytes)):

                # type is str and item_size is given for column
                if (
                    min_itemsize is not None
                    and isinstance(min_itemsize, dict)
                    and key in min_itemsize
                ):
                    attrs[key] = attr.ib(
                        type=type(df.loc[0, key]),
                        metadata={"itemsize": min_itemsize[key]},
                    )

                # type is str and item_size is the same for all
                elif min_itemsize is not None and isinstance(
                    min_itemsize, int
                ):
                    attrs[key] = attr.ib(
                        type=type(df.loc[0, key]),
                        metadata={"itemsize": min_itemsize},
                    )

                # type str find max length of str in given column
                else:
                    # type is str and item_size is not given -> take
                    # max length in the column or raise ValueError if
                    # the max length is zero
                    max_str_len = df.loc[:, key].str.len().max()
                    if max_str_len == 0:
                        raise ValueError(
                            "String length co not be deduced "
                            f"from empty column {key}"
                        )
                    attrs[key] = attr.ib(
                        type=type(df.loc[0, key]),
                        metadata={"itemsize": max_str_len},
                    )

            # object is actually a Dataclass
            elif attr.has(df.loc[0, key]):
                attrs[key] = attr.ib(type=type(df.loc[0, key]))
            else:
                raise TypeError(
                    f"Unsupported type: {type(df.loc[0, key]).__name__}, "
                    "make a coffee to the maintainer and ask for help."
                )
        else:
            raise TypeError(
                f"Unsupported type: {type(df.loc[0, key]).__name__}, "
                "make a coffee to the maintainer and ask for help."
            )
    return attr.make_class(name=name, attrs=attrs, auto_attribs=True)
