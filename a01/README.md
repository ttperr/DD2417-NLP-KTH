# DD2417 Language Engineering - Assignment 1

## Mandatory questions

> 1. Find all the syntax trees derivable from the sentence *“He built the box with a hammer in the yard behind the house”*

![Syntax tree 1](graph/1-syntax-tree-1.svg)

![Syntax tree 2](graph/1-syntax-tree-2.svg)

![Syntax tree 3](graph/1-syntax-tree-3.svg)

![Syntax tree 4](graph/1-syntax-tree-4.svg)

> Convert the following grammar into a weakly equivalent Chomsky normal form grammar.

```grammar
S   → NP VP         Noun → faces
NP  → Det Adj Noun  Noun → suspect
NP  → Adj Noun      Noun → pressure
NP  → Det Noun       Adj → increased
NP  → Noun           Adj → suspect
VP  → V NP             V → faces
Det → the              V → increased
```

become

```grammar
S   → NP VP         NP → faces
NP  → DA NP         NP → suspect
DA  → Det Adj      Adj → suspect
NP  → Adj NP        NP → pressure
NP  → Det NP       Adj → increased
VP  → V NP           V → faces
Det → the            V → increased
```

> Using your new grammar, use the CKY algorithm to parse the sentence “the suspect faces increased pressure”. Show your completed parse table as result.

| the | suspect | faces | increased | pressure |
| --- | ------- | ----- | --------- | -------- |
| Det | NP DA   | NP    |           | S S      |
|     | Adj NP  | NP    |           | S S      |
|     |         | V NP  |           | VP S     |
|     |         |       | Adj V     | NP VP    |
|     |         |       |           | NP       |

And if we extract all the trees we get:

![CKY tree 1](graph/2-CKY-tree-1.svg)

![CKY tree 2](graph/2-CKY-tree-2.svg)

> Each of these dependency trees has one edge which is incorrect. Decide which one, and explain how it should be drawn instead.

In the **a** tree, the root edge is incorrect. It should be drawn from **ROOT** to **wanted** and not the opposite.

In the **b** tree, the edge from **lasted** to **The** is incorrect. It should be drawn from **meeting** to **The**.

In the **c** tree, the edge from **gave** to **brother** is incorrect. It should be drawn from **gave** to **book**.

> Below is a correct dependency tree, but with the labels missing. For each of the labels 1–7, determine the appropriate relation label.

1. root
2. nsubj
3. amod
4. det
5. advmod
6. obj
7. det
