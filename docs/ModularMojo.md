# Modular Mojo notes (from docs/agent-reference/modular/mojo)

## Overview

- `docs/agent-reference/modular` is the full Modular Platform repo; Mojo lives under `mojo/` and is part of the broader MAX/Mojo platform. (modular/README.md)
- The `mojo/` subtree includes the Mojo standard library, language proposals, docs, examples, and tests. (mojo/README.md)

## `mojo/` layout (selected)

- `stdlib/`: Mojo standard library sources; the reference docs are generated from `stdlib/std`. (mojo/docs/README.md)
- `docs/`: Markdown/MDX content for https://docs.modular.com/mojo (excluding `mojo` CLI pages and stdlib reference). (mojo/docs/README.md)
- `examples/`: language and standard library usage examples. (mojo/README.md)
- `proposals/`: design and evolution proposals for the language. (mojo/README.md)
- `integration-test/`: integration tests for Mojo. (mojo/README.md)
- `python/`: Python-related tooling/bindings used by the Mojo toolchain. (mojo/README.md)

## Documentation notes

- The Mojo documentation is maintained in `mojo/docs/` and published to the public docs site. (mojo/docs/README.md)
- The Mojo standard library reference is generated from source under `stdlib/std`, not stored directly in `mojo/docs/`. (mojo/docs/README.md)

## Practical takeaways for Strut

- `mojo/stdlib` and `mojo/docs` are the canonical references for language/library behavior and documentation style.
- `mojo/examples` can serve as reference implementations and patterns for performance or API usage.
