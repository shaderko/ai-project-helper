# AI Project Helper

This project is aimed to help LLMs understand code you have on your local machine, using rag retrieval.

## Installation

Create a new pip environment (optional, but recomended)

```bash
pipenv shell
```

Install dependecies from requirement.txt

```bash
pip install -r requirements.txt
```

## Usage

Run from cli, using already stored data.

```bash
python rag_agent.py
```

If you want to start for a new project pass it as --doc-dir

```bash
python rag_agent.py --doc-dir F:/my/massive/directory/of/code
```

You can also change the persistent directory if you want to.

```bash
python rag_agent.py --persist-dir D:/stored/on/a/different/drive
```

For multiple projects you can change the persist dir if you already initialized it but there is not yet another way of changing projects.

## Recommendations

Current model in the python file is qwen3-32:4b which is just a larger context qwen3:4b. To create larger context model in ollama, run the ollama model you want to adjust with

```bash
ollama run qwen3:4b
```

Once the chat is open, adjust context with the following command

```bash
/set num_ctx 32000
```

And save the model

```bash
/save qwen3-32:4b
```

And thats it, you can now use this model with higher context size, but be aware that higher context size needs more Vram, if it doesn't have enough Vram it will offload on to the CPU, which gets way slower.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[APACHE](https://choosealicense.com/licenses/apache-2.0/)
