from typing import Protocol, runtime_checkable


@runtime_checkable
class RoutableEntity(Protocol):
    """An entity the routing loop can match against user messages.

    *param_name* declares which tool parameter this entity maps to
    (e.g. ``"product_key"``).  The routing loop uses it to inject
    the entity's *key* into tool arguments when a match is found.
    """

    key: str
    short_name: str
    param_name: str
