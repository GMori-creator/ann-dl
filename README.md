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
    index.md                   # visão geral: equipe, dataset, as 3 entregas
    eda/{index.md,code/,figures/}
    classification/{index.md,code/,figures/}
    regression/{index.md,code/,figures/}
    generative/{index.md,code/,figures/}
  examples/                    # exemplos de uso do menu (pode ser removido)
```

Os slugs de `exercises/` e `projects/` são fixos e casam com o site da disciplina. Não os
renomeie.

## As entregas de 2026.2

**Exercícios** (individuais, 25% cada dentro dos 40% de exercícios da nota individual):
Data (01/09), Perceptron (10/09), MLP (22/09), VAE (20/10).

**Projeto** (equipe, um único dataset em três entregas): EDA (17/09, 20%),
Classificação **ou** Regressão (05/11, 60%), Generativo (20/11, 20%). A prova de projeto
(19/11) limita a nota de equipe — `Equipe = min(Projeto, Prova)`.

O template traz as pastas de classificação e regressão; apague a que a equipe não escolher,
da pasta e da `nav`.

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

O workflow em [.github/workflows/main.yaml](.github/workflows/main.yaml) roda
`mkdocs gh-deploy --force` a cada push na `main`: ele constrói o HTML, empurra para a branch
`gh-pages`, e é essa branch que o GitHub Pages serve.

Configuração inicial, uma vez:

1. **Se você forkou**, habilite os workflows na aba **Actions** (forks vêm com o Actions
   desligado). Usando *Use this template* isso não é necessário.
2. Troque no [mkdocs.yml](mkdocs.yml) todas as linhas marcadas com `# TROCAR`
   (`grep -n TROCAR mkdocs.yml`).
3. **Settings → Actions → General → Workflow permissions** → **Read and write permissions**.
   Sem isso o CI falha com `Permission denied to github-actions[bot]`.
4. Dê o primeiro push e espere o run terminar — é ele que cria a branch `gh-pages`.
5. **Settings → Pages** → *Deploy from a branch* → branch **`gh-pages`**, pasta **`/ (root)`**.

O passo a passo com as telas está em
[Como usar este template → Publicação no GitHub Pages](docs/template/index.md).

Antes de dar push, valide localmente — o CI publica mesmo com avisos, o modo estrito não:

```shell
mkdocs build --strict
```

Para publicar manualmente, sem passar pelo CI:

```shell
mkdocs gh-deploy
```

## Prazo

O prazo de uma entrega é o **timestamp do último commit que toca a pasta daquela entrega**
— não a hora do formulário nem a da publicação. Commite progressivamente.
