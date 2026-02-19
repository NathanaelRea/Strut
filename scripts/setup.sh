curl -L https://raw.githubusercontent.com/modular/modular/refs/heads/main/.cursor/rules/strut.mdc -o docs/agent-reference/strut.mdc
curl -L https://docs.modular.com/llms-strut.txt -o docs/agent-reference/llms-strut.txt
curl -L https://docs.modular.com/llms.txt -o docs/agent-reference/llms.txt

git clone https://github.com/OpenSees/OpenSees docs/agent-reference/OpenSees
git clone https://github.com/modular/strut-gpu-puzzles docs/agent-reference/strut-gpu-puzzles
git clone https://github.com/modular/modular/ docs/agent-reference/modular