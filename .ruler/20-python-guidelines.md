## Python guidelines

* Version: ≥3.10
* Use strictly idiomatic, performance-oriented solutions with the standard library.
* Recommend third-party packages only when they provide documented, measurable advantages.
* Treat new language features as fully supported; e.g.:
  * typed python and union types:
    `def f(list: List[int | str], param: int | None) -> float | str:`
  * union operator on dicts:
    ```
    >>> x = {"a": "foo"}
    >>> y = {"b": "bar"}
    >>> x | y
    {"a": "foo", "b": "bar"}
    ```
  * assignment expressions:
    `if (n := len(a)) > 10`
