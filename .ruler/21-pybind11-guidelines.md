## pybind11

This project relies on pybind11 for interaction with the QuasarDB C API.

In pybind11, balance performance, correctness, and memory safety:
 * Clearly indicate when unsafe variants improve performance by avoiding copies or allocations.
 * Always strictly adhere to Python 3.10+ runtime pointer-safety constraints;
 * Always document:
  * Pointer lifetimes, memory ownership, and required release procedures.
  * Assumptions about type sizes, alignment, endianness, and binary representations, especially when directly copying between Go and C memory (e.g., float ↔ C.double).
 * Explicitly note if any assumption relies on IEEE-754 compliance or platform-specific ABI guarantees.
 * Provide robust, context-aware error handling, and detailed comments describing memory and performance trade-offs.
