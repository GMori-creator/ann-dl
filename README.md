# Template de entregas — Redes Neurais Artificiais & Deep Learning

Template de site (MkDocs Material + GitHub Pages) para as entregas da disciplina,
edição **2026.2** — [enunciados](https://insper.github.io/ann-dl/2026.2/).

Cada entrega é um item de menu, e o alvo do item pode ser um relatório em Markdown, um
notebook `.ipynb` ou um link do Google Colab. Os três casos estão demonstrados na seção
**Exemplos de uso** do site.

## Estrutura

```
docs/
  index.md                     # capa: grupo e status das entregas
  template/index.md            # como usar este template
  exercises/
    data/{index.md,code/,figures/}
    perceptron/{index.md,code/,figures/}
    mlp/{index.md,code/,figures/}
    vae/{index.md,code/,figures/}
  projects/
    classification/{index.md,code/,figures/}
    regression/{index.md,code/,figures/}
    generative/{index.md,code/,figures/}
  examples/                    # exemplos de uso do menu (pode ser removido)
```

Os slugs de `exercises/` e `projects/` são fixos e casam com o site da disciplina. Não os
renomeie.

## Setup

```shell
python3 -m venv env
source ./env/bin/activate          # Windows: .\env\Scripts\activate
python3 -m pip install -r requirements.txt --upgrade
```

## Rodando localmente

```shell
mkdocs serve -o
```

## Publicação

O workflow em [.github/workflows/main.yaml](.github/workflows/main.yaml) publica o site a
cada push na `main`. Antes do primeiro push, ajuste `site_url`, `repo_url` e `repo_name` no
[mkdocs.yml](mkdocs.yml) para o seu repositório.

Para publicar manualmente:

```shell
mkdocs gh-deploy
```

## Prazo

O prazo de uma entrega é o **timestamp do último commit que toca a pasta daquela entrega**
— não a hora do formulário nem a da publicação. Commite progressivamente.
