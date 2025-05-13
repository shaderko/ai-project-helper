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

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[APACHE](https://choosealicense.com/licenses/apache-2.0/)
