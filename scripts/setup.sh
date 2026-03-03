curl -L https://raw.githubusercontent.com/modular/modular/refs/heads/main/.cursor/rules/mojo.mdc -o docs/agent-reference/mojo.mdc
curl -L https://docs.modular.com/llms-mojo.txt -o docs/agent-reference/llms-mojo.txt
curl -L https://docs.modular.com/llms.txt -o docs/agent-reference/llms.txt

git clone https://github.com/OpenSees/OpenSees docs/agent-reference/OpenSees
git clone https://github.com/modular/modular/ docs/agent-reference/modular

rm -rf docs/agent-reference/OpenSees/.git
rm -rf docs/agent-reference/modular/.git

python3 setup-opensees-docs.py