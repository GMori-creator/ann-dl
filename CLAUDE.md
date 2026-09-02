# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este repositório

Um **template de site MkDocs** (Material + GitHub Pages) que alunos forkam para entregar os
exercícios e projetos da disciplina de Redes Neurais Artificiais & Deep Learning, edição
**2026.2**. Não há aplicação nem biblioteca aqui — o produto é o site em `docs/`.

Os enunciados que este template atende ficam em <https://insper.github.io/ann-dl/2026.2/>;
as regras de entrega, em <https://insper.github.io/ann-dl/2026.2/exercises/submission/>.

## Comandos

```shell
python3 -m venv env && source ./env/bin/activate
python3 -m pip install -r requirements.txt --upgrade

mkdocs serve -o      # preview local com reload
mkdocs build --strict # validação: falha em link quebrado ou snippet inexistente
mkdocs gh-deploy     # publicação manual (o CI já faz isso a cada push na main)
```

Não há testes. **`mkdocs build --strict` é o teste** — use-o antes de considerar qualquer
alteração pronta. Ele já roda limpo, com zero warnings; mantenha assim.

## Arquitetura

### O contrato de pastas é imposto de fora

Os slugs em `docs/exercises/` (`data`, `perceptron`, `mlp`, `vae`) e em `docs/projects/`
(`classification`, `regression`, `generative`) são fixados pelas *submission guidelines* da
disciplina, assim como o caminho `docs/<seção>/<slug>/index.md` e as subpastas `code/` e
`figures/`. **Renomear qualquer um deles quebra a correção**, não só o site. A exceção é
`projects/eda/`: o overview de 2026.2 cobra a entrega de EDA, mas a disciplina ainda não
publicou uma página com esse slug — se ela aparecer com outro nome, alinhe.

Todo relatório abre com *front matter* obrigatório: `exercise:` (ou `project:`) e `ai_use:`.

### Exercícios são 4; projeto é 1 com 3 entregas

Fácil de errar: `docs/projects/` **não** contém três projetos independentes. É um único
projeto de equipe, sobre um único dataset, entregue em três partes — EDA (17/09, 20%),
Classificação **ou** Regressão (05/11, 60%), Generativo (20/11, 20%). O template traz as
pastas de classificação e regressão porque a equipe escolhe uma; as duas aparecem na `nav`
com um comentário mandando apagar a que sobrar.

`docs/projects/index.md` é a página de visão geral (equipe, dataset, registro de decisões) e
funciona como *section index* — daí o `navigation.indexes` nas features do tema.

Datas e pesos vivem em <https://insper.github.io/ann-dl/2026.2/> e estão replicados em
`docs/index.md`, `docs/projects/index.md` e no cabeçalho de cada entrega. Ao atualizá-los,
atualize os quatro lugares.

### Uma entrega, um item de menu

O `nav` do `mkdocs.yml` tem um item por entrega, e o alvo pode ser um `.md`, um `.ipynb`
(renderizado pelo `mkdocs-jupyter`) ou uma URL absoluta do Colab. `docs/examples/index.md`
demonstra os três e é a página a atualizar quando esse mecanismo mudar.

### Código não é copiado, é incluído

`docs/exercises/data/index.md` é o modelo preenchido: puxa o script real de `code/` via
`--8<--` do `pymdownx.snippets`, configurado com `base_path: [.]` (raiz do repositório) e
`check_paths: true` — por isso um caminho errado derruba o build. Ao escrever `--8<--` como
exemplo dentro de uma página, escape com `;--8<--`, senão o preprocessador o executa mesmo
dentro de bloco de código.

### Decisões de configuração que não são óbvias

- `mkdocs-jupyter` recebe `include: ["*.ipynb"]` porque, por padrão, ele converteria também
  os `.py` de `code/` em páginas do site.
- `not_in_nav` cobre `code/*` para que esses arquivos fiquem no repositório sem virar item
  de navegação nem gerar warning em `--strict`.
- `javascripts/mathjax.js` precisa vir **antes** do CDN do MathJax em `extra_javascript`:
  a configuração é lida no carregamento da biblioteca.
- Plugins deliberadamente ausentes: `markdown-exec` (removido a pedido), `mkdocs-badges`,
  `render_swagger` e `git-committers` (exigia token de API no CI). Não os reintroduza sem
  motivo explícito.

### Publicação

`.github/workflows/main.yaml` roda `mkdocs gh-deploy --force` em todo push na `main`. O
prazo de uma entrega é o timestamp do último commit que toca a pasta dela — o que torna o
histórico do Git parte da avaliação, e commits que reescrevem histórico, arriscados.

## Ao editar este template

Os placeholders (`usuario/ann-dl`, `Seu Nome`, `fig01-exemplo.svg`, tabelas em branco) são
intencionais: o aluno os substitui. Não os "conserte" com dados fictícios plausíveis. O
conteúdo do site é em **português**; os slugs e os títulos de seção que espelham o enunciado
permanecem em inglês.
