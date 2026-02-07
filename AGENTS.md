# Strut

A adaptation of the popular finite element and earthquake analysis tool `OpenSees` into the high performance, gpu-enhanced programming language mojo.

## `docs/agent-reference/` resources

The `docs/agent-reference/` directory is collection of documents for development in mojo.

Official documentation:

- [mojo.mdc](./docs/agent-reference/mojo.mdc)
  - small file with best practices for mojo code. **you should always read this file** if working with anything mojo
- [llms.txt](./docs/agent-reference/llms.txt)
  - medium size file (2.5k lines) with urls to docs for every mojo specific keyword
- [llms-mojo.txt](./docs/agent-reference/llms-mojo.txt)
  - very large file (75k+ lines) with extensive docs for mojo **never read the full file**, only query/grep it for line ranges and read those

Cloned repos:

- [OpenSees](./docs/agent-reference/OpenSees/)
  - the full repository of `OpenSees` written in c++
  - this is the source of truth for implementation and benchmarks
- [modular](./docs/agent-reference/modular/)
  - the full repository of `Modular`
  - this is the source of truth for [mojo](./docs/agent-reference/modular/mojo/), contains all the code, docs, implementations, and examples
- [mojo-gpu-puzzles](./docs/agent-reference/mojo-gpu-puzzles/)
  - some possible useful examples for using the gpu with mojo, if needed look through the [solutions](./docs/agent-reference/mojo-gpu-puzzles/solutions/)

## Testing

- [testing.md](./docs/testing.md)

## A note to the Agent

This is a living document, and so if something takes a long time to figure out, write a very short summary here or in a nested md file inside [docs/](./docs/) so we can develop faster in the future. This also includes when researching the documentation or cloned repos above.
