## Performance

Performance is critical. Code will run trillions of times per day in a high-performance timeseries database.

- Avoid copies of arrays, but instead prefer zero-copy pointer-passing where possible
- Explicitly document trade-offs (complexity, maintainability, readability vs performance).
- Recommend explicitly when each approach should be chosen.
- Provide concise, actionable examples of recommended performance techniques.

## Function Documentation Style

When documenting functions:

- Use concise, high-signal comments that explain *why* and *how* rather than repeating *what* the code already shows.
- Use pydoc-style function docs that clearly document all arguments, e.g.:
  ```
  def first(xs: List[int | str]) -> int | str:
     """
     Returns first element of the list.

     Parameters:
     -----------

     xs : List[int | str]
       The list to process
     """
     return xs[0]
  ```
- For non-trivial functions, include bullet lists for key sections in the pydoc function docs:
  - `Decision rationale:` followed by short bullets describing choices made.
  - `Key assumptions:` bullets stating the required preconditions or invariants.
  - `Performance trade-offs:` bullets outlining costs versus benefits.
- For non-trivial functions, provide an inline usage example in the pydoc function docs.
- For inline comments, use _only_ single-line comments starting with `#`.
- Avoid trivial comments. Focus on explaining design choices, trade-offs and assumptions.
- Optimize for clarity and LLM readability. Never use emojis.

## Helper Function Guidelines

Create small, composable helper functions where practical.
