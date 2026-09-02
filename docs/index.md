# Entregas — Redes Neurais Artificiais & Deep Learning

???+ info inline end "Edição"

    **2026.2**

    [Enunciados :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/){:target='_blank'}

Este site é o **portfólio** das entregas da disciplina. Ele cresce ao longo do semestre:
cada exercício e cada projeto vira um item de menu, e o repositório que o gera é parte
da avaliação — o professor lê o site publicado **e** o repositório (Markdown, código e
histórico do Git).

## Aluno / Grupo

| Nome | E-mail | GitHub |
|------|--------|--------|
| João da Silva | joaods@al.insper.edu.br | [@joaods](https://github.com/joaods) |
| Maria Oliveira | mariao@al.insper.edu.br | [@mariao](https://github.com/mariao) |

!!! tip "Como usar este template"

    Este é um **bloco de notas versionado**: registre o que foi feito, o que falta e as
    decisões tomadas, commitando a cada avanço. O prazo de uma entrega é o *timestamp do
    último commit que toca a pasta daquela entrega* — não a hora do formulário nem a da
    publicação no Pages.

    Comece por [Como usar este template](template/index.md).

## Status das entregas

A nota individual e a de equipe são **independentes** — as duas precisam chegar a 5,0.

### Exercícios — individual, 40% da nota individual

| # | Entrega | Data | Peso | Status |
|---|---------|------|------|--------|
| 1 | [Data](exercises/data/index.md) | 01/09 | 25% | :material-checkbox-blank-outline: |
| 2 | [Perceptron](exercises/perceptron/index.md) | 10/09 | 25% | :material-checkbox-blank-outline: |
| 3 | [MLP](exercises/mlp/index.md) | 22/09 | 25% | :material-checkbox-blank-outline: |
| 4 | [VAE](exercises/vae/index.md) | 20/10 | 25% | :material-checkbox-blank-outline: |

Os outros 60% da nota individual vêm da prova final (24/11).

### [Projeto](projects/index.md) — equipe

Um projeto, um dataset, três entregas:

| # | Entrega | Data | Peso | Status |
|---|---------|------|------|--------|
| 1 | [EDA](projects/eda/index.md) | 17/09 | 20% | :material-checkbox-blank-outline: |
| 2 | [Classificação](projects/classification/index.md) **ou** [Regressão](projects/regression/index.md) | 05/11 | 60% | :material-checkbox-blank-outline: |
| 3 | [Generativo](projects/generative/index.md) | 20/11 | 20% | :material-checkbox-blank-outline: |

A prova de projeto (19/11) **limita** a nota de equipe em vez de somar a ela:
$\text{Equipe} = \min(\text{Projeto},\ \text{Prova de Projeto})$.

## Checklist antes de cada entrega

- [ ] Repositório **público** e o GitHub Pages construindo sem erro.
- [ ] Caminho correto: `docs/exercises/<slug>/index.md` (ou `docs/projects/<slug>/index.md`).
- [ ] *Front matter* com `exercise:` (ou `project:`) e `ai_use:` preenchidos.
- [ ] Títulos espelhando a estrutura do enunciado (`## Exercise N`, `### A`, `### B`, ...).
- [ ] Figuras commitadas em `figures/`, numeradas e exibidas no relatório.
- [ ] Scripts como arquivos reais em `code/`, referenciados via `--8<--`.
- [ ] Tabela **Results summary** completa, sem linhas em branco.
- [ ] Último commit anterior ao prazo.

!!! danger "Defesa oral"

    **Todas** as notas da disciplina estão sujeitas a defesa oral. Resultado negativo na
    defesa **zera** aquela nota. Escreva relatórios que você consiga sustentar oralmente —
    o que inclui entender cada linha do código que está no repositório.

!!! danger "Uso de IA"

    O campo `ai_use` é **obrigatório** em toda entrega. Colaborar com IA é permitido;
    não declarar o uso, não. Descreva o que foi gerado, revisado ou depurado com apoio de
    IA — ou escreva `"none"`.
